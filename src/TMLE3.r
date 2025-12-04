options(repos = c(CRAN = "https://cloud.r-project.org"))
required_pkgs <- c("here", "tidyverse", "data.table", "remotes", "haldensify", "Rsolnp", "xgboost", "nnls")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}
github_pkgs <- c("tlverse/sl3", "tlverse/tmle3", "tlverse/tmle3shift")
for (pkg in github_pkgs) {
  pkg_name <- unlist(strsplit(pkg, "/"))[2]
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    remotes::install_github(pkg)
  }
}

library(here)
library(tidyverse)

# Can change the csv used to other data as needed
bf <- read_csv(here("Outputs", "normalized_anomaly_9k.csv"))
bf

library(data.table)
library(sl3)
library(tmle3)
library(tmle3shift)

# Note: Remove solar_modulation and volcanic_forcing from the list if using the 742k dataset
data <- bf [,c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation", "solar_modulation", "volcanic_forcing", "co2_ppm", "anomaly"), with = FALSE]
data <- as.data.table(data)
node_list <- list(
  W = c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation", "solar_modulation", "volcanic_forcing"),
  A = "co2_ppm",
  Y = "anomaly"
)
head(data)

# Change shift_val to desired delta
tmle_spec <- tmle_shift(
  shift_val = 0.05,
  shift_fxn = shift_additive,
  shift_fxn_inv = shift_additive_inv
)

sl3_list_learners("density")

# Note: if using 742k dataset, can instead use n_bins = 7, lambda_seq = exp(seq(-1, -8, length = 75))
haldensify_lrnr <- Lrnr_haldensify$new(
  n_bins = 5, grid_type = "equal_mass",
  lambda_seq = exp(seq(-1, -7, length = 50))
)
hse_lrnr <- Lrnr_density_semiparametric$new(mean_learner = Lrnr_glm$new())
mvd_lrnr <- Lrnr_density_semiparametric$new(
  mean_learner = Lrnr_glm$new(),
  var_learner = Lrnr_mean$new()
)
sl_lrn_dens <- Lrnr_sl$new(
  learners = list(haldensify_lrnr, hse_lrnr, mvd_lrnr),
  metalearner = Lrnr_solnp_density$new()
)

mean_lrnr <- Lrnr_mean$new()
glm_lrnr <- Lrnr_glm$new()
xgb_lrnr <- Lrnr_xgboost$new()
sl_lrn <- Lrnr_sl$new(
  learners = list(mean_lrnr, glm_lrnr, xgb_lrnr),
  metalearner = Lrnr_nnls$new()
)

Q_learner <- sl_lrn
g_learner <- sl_lrn_dens
learner_list <- list(Y = Q_learner, A = g_learner)

# Note: This part usually takes a while to run
# and if delta is too high, a large percent of the predictions get trimmed
tmle_fit <- tmle3(tmle_spec, data, node_list, learner_list)

tmle_fit
