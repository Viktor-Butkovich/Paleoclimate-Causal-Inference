# Reference: https://journals.sagepub.com/doi/10.1177/26320843231176662 

# Install needed packages
options(repos = c(CRAN = "https://cloud.r-project.org"))
required_pkgs <- c("here", "tidyverse", "data.table", "nloptr", "dbarts")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(here)
library(tidyverse)

# Can change the csv as needed; if running the 742k dataset, use read_csv(here("Outputs", "normalized_anomaly_742k.csv"))
bf <- read_csv(here("Outputs", "normalized_anomaly_9k.csv"))
bf

library(data.table)

# Note: Remove solar_modulation and volcanic_forcing from the list if using the 742k dataset; remove "volcanic_forcing" to run old version
ObsData <- data.frame()
ObsData <- as.data.table(bf)
ObsData <- bf [,c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation", "solar_modulation", "volcanic_forcing"), with = FALSE]          
ObsData$A <- bf$"co2_ppm"
ObsData$Y <- bf$"anomaly"            

head(ObsData)
names(ObsData)

# Box 1 - Transforming outcome Y to the scale [0, 1]  
min.Y <- min(ObsData$Y)
max.Y <- max(ObsData$Y)
ObsData$Y.bounded <- (ObsData$Y-min.Y) / (max.Y-min.Y)

# Check range of transformed outcome data
summary(ObsData$Y.bounded)


# Box 2 - Fitting initial outcome model using SL pred for obs treatmnt assign
library(SuperLearner)
library(tmle)
# note: can probs change seed val if want
set.seed(123)
ObsData.noY <- dplyr::select(ObsData, !c(Y, Y.bounded))
Y.fit.sl <- SuperLearner(Y=ObsData$Y.bounded, X=ObsData.noY, cvControl = list(V=3), SL.library=c("SL.glm", "tmle.SL.dbarts2", "SL.glmnet"), method="method.CC_nloglik", family="gaussian")
ObsData$Pred.initialQ0 <- Y.fit.sl$SL.predict

# check range of new pred outcomes
summary(ObsData$Pred.initialQ0)


#  Box 3a - counterfac outcome pred, assign all trtmnts A = 1
ObsData.noYA1 <- ObsData.noY
ObsData.noYA1$A <- 1
ObsData$Pred.initialQ0.Y1 <- SuperLearner(Y=ObsData$Y.bounded, X=ObsData.noY, cvControl = list(V=3), SL.library=c("SL.glm", "tmle.SL.dbarts2", "SL.glmnet"), method="method.CC_nloglik", family="gaussian", newX = ObsData.noYA1)$SL.predict
summary (ObsData$Pred.initialQ0.Y1)

# Box 3b - making pred for all trmtnts A = 0
ObsData.noYA0 <- ObsData.noY
ObsData.noYA0$A <- 0
ObsData$Pred.initialQ0.Y0 <- SuperLearner(Y=ObsData$Y.bounded, X=ObsData.noY, cvControl = list(V=3), SL.library = c("SL.glm", "tmle.SL.dbarts2", "SL.glmnet"), method="method.CC_nloglik", family="gaussian", newX = ObsData.noYA0)$SL.predict
summary(ObsData$Pred.initialQ0.Y0)

# Box 4 - calc crude ATE estimate
ObsData$Pred.TE <- ObsData$Pred.initialQ0.Y1 - ObsData$Pred.initialQ0.Y0
mean(ObsData$Pred.TE)


# Box 5 - construct prop score models and extract prop scores through pred
library(SuperLearner)
library(tmle)
set.seed(124) 
ObsData.noYA <- dplyr::select(ObsData, !c(Y, Y.bounded, A, Pred.initialQ0, Pred.initialQ0.Y1, Pred.initialQ0.Y0, Pred.TE))
PS.fit.SL <- SuperLearner(Y=ObsData$A, X=ObsData.noYA, cvControl = list(V=3), SL.library=c("SL.glm", "SL.gam", "tmle.SL.dbarts.k.5"), method="method.CC_nloglik", family="binomial")
ObsData$PS.SL <- PS.fit.SL$SL.predict
summary(ObsData$PS.SL)


# Box 6 - using prop scores to calc clever covar
ObsData$H.A1L <- (ObsData$A) / ObsData$PS.SL
summary(ObsData$H.A1L)
ObsData$H.A0L <- (1-ObsData$A) / (1-ObsData$PS.SL)
summary(ObsData$H.A0L)
ObsData$H.AL <- ObsData$H.A1L - ObsData$H.A0L
summary(ObsData$H.AL)


# Box 7 - using clever covar and init. pred to est fluc parameter epsilon
eps_mod <- glm(Y.bounded ~ -1 + H.A1L + H.A0L + offset(qlogis(Pred.initialQ0)), family = "binomial", data = ObsData)
epsilon <- coef(eps_mod)
epsilon["H.A1L"]
epsilon["H.A0L"]


# Box 8 - updating pred. counterfac outcomes using clever cov and fluc param
ObsData$Pred.updateQ1.Y1 <- plogis(qlogis(ObsData$Pred.initialQ0.Y1) + epsilon["H.A1L"] *ObsData$H.A1L)
ObsData$Pred.updateQ1.Y0 <- plogis(qlogis(ObsData$Pred.initialQ0.Y0) + epsilon["H.A0L"]*ObsData$H.A0L)
summary(ObsData$Pred.updateQ1.Y1)
summary(ObsData$Pred.updateQ1.Y0)

# Box 9 - calc ATE by avg dif betwn updated trtd and untrtd counterfac outcm
ATE.TMLE.bounded.vector <- ObsData$Pred.updateQ1.Y1 - ObsData$Pred.updateQ1.Y0
summary(ATE.TMLE.bounded.vector)
ATE.TMLE.bounded <- mean(ATE.TMLE.bounded.vector, na.rm = TRUE)
ATE.TMLE.bounded


# Box 10 - transfrm ATE back to org outcome scale
ATE.TMLE <- (max.Y-min.Y)*ATE.TMLE.bounded
ATE.TMLE


# Box 11 - CI estim using eff influ curve
#transf pred outcomes back to orgin scale
ObsData$Pred.updateQ1.Y1.rescaled <- (max.Y-min.Y)*ObsData$Pred.updateQ1.Y1 + min.Y
ObsData$Pred.updateQ1.Y0.rescaled <- (max.Y-min.Y)*ObsData$Pred.updateQ1.Y0 + min.Y

EY1_TMLE1 <- mean(ObsData$Pred.updateQ1.Y1.rescaled, na.rm = TRUE)
EY0_TMLE1 <- mean(ObsData$Pred.updateQ1.Y0.rescaled, na.rm = TRUE)

# ATE eff influ curve
D1 <- ObsData$A/ObsData$PS.SL*(ObsData$Y - ObsData$Pred.updateQ1.Y1.rescaled) + ObsData$Pred.updateQ1.Y1.rescaled - EY1_TMLE1
D0 <- (1 - ObsData$A)/(1 - ObsData$PS.SL)*(ObsData$Y - ObsData$Pred.updateQ1.Y0.rescaled) + ObsData$Pred.updateQ1.Y0.rescaled - EY0_TMLE1
EIC <- D1 - D0

# ATE var
n <- nrow(ObsData)
varHat.IC <- var(EIC, na.rm = TRUE) / n

# ATE 95% CI
ATE.TMLE.CI <- c(ATE.TMLE - 1.96*sqrt(varHat.IC), ATE.TMLE + 1.96*sqrt(varHat.IC))
ATE.TMLE.CI


# Box 12 -  estm ATE using tmle package 
ObsData <- as.data.table(ObsData)

# discretize A; change cut-off to test different vals
ObsData[, A_bin := ifelse(A > median(A, na.rm = TRUE), 1, 0)]

# clean up data in case there are na
W <- dplyr::select(ObsData, !c(Y, Y.bounded, A, A_bin))
W <- as.data.frame(W)
W_clean <- W[, colSums(is.na(W)) < nrow(W), drop = FALSE]
tmle_data <- cbind(
  Y = ObsData$Y.bounded,
  A = ObsData$A_bin,
  W_clean
)
tmle_data_nna <- tmle_data[complete.cases(tmle_data), ]
if(nrow(tmle_data_nna) == 0){
  stop("error")
}
Y_nna <- tmle_data_nna$Y
A_nna <- tmle_data_nna$A
W_nna <- as.data.frame(tmle_data_nna[, setdiff(names(tmle_data_nna), c("Y","A"))])
cat("Rows:", nrow(Y_nna), "W Columns:", ncol(W_nna), "\n")

# tmle
set.seed(123)
TMLE_PKG <- tmle(
  Y = Y_nna,
  A = A_nna,
  W = W_nna,
  family = "gaussian"
)

TMLE_PKG


TMLE_PKG_ATE_tr <- TMLE_PKG$estimates$ATE$psi
TMLE_PKG_ATE_tr


# transform back the ATE estim using bds of orig Y
TMLE_PKG_ATE <- (max.Y - min.Y)*TMLE_PKG_ATE_tr
TMLE_PKG_ATE


TMLE_PKG_CI <- (max.Y-min.Y)*TMLE_PKG$estimates$ATE$CI
TMLE_PKG_CI


# visual for comparing the ATEs
library(ggplot2)

pEY1 <- TMLE_PKG$estimates$EY1$psi
pEY0 <- TMLE_PKG$estimates$EY0$psi
pATE <- TMLE_PKG$estimates$ATE$psi

pd <- data.frame(
  Exposure = c("A=0 (control)", "A=1 (treated)"),
  Outcome = c(pEY0, pEY1)
)

ggplot(pd, aes(x = Exposure, y = Outcome, fill = Exposure)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(Outcome, 2)), vjust = -0.5) +
  ylim(0, max(pEY1, pEY0) * 1.2) +
  theme_minimal() +
  labs(title = paste0("Average Treatment Effect: ", round(pATE, 2)))
