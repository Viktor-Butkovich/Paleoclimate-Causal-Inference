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
library(data.table)
library(sl3)
library(tmle3)
library(tmle3shift)


make_learners <- function(dataset = c("9k", "742k")) {
    dataset <- match.arg(dataset)

    # density learners
    if (dataset == "9k") {
        haldensify_lrnr <- Lrnr_haldensify$new(
            n_bins = 5, grid_type = "equal_mass",
            lambda_seq = exp(seq(-1, -7, length = 50))
        )
    } else if (dataset == "742k") {
        haldensify_lrnr <- Lrnr_haldensify$new(
            n_bins = 7, grid_type = "equal_mass",
            lambda_seq = exp(seq(-1, -8, length = 75))
        )
    }

    hse_lrnr <- Lrnr_density_semiparametric$new(mean_learner = Lrnr_glm$new())
    mvd_lrnr <- Lrnr_density_semiparametric$new(
        mean_learner = Lrnr_glm$new(),
        var_learner = Lrnr_mean$new()
    )
    sl_lrn_dens <- Lrnr_sl$new(
        learners = list(haldensify_lrnr, hse_lrnr, mvd_lrnr),
        metalearner = Lrnr_solnp_density$new()
    )

    # outcome learners
    mean_lrnr <- Lrnr_mean$new()
    glm_lrnr <- Lrnr_glm$new()
    xgb_lrnr <- Lrnr_xgboost$new()
    sl_lrn <- Lrnr_sl$new(
        learners = list(mean_lrnr, glm_lrnr, xgb_lrnr),
        metalearner = Lrnr_nnls$new()
    )

    Q_learner <- sl_lrn
    g_learner <- sl_lrn_dens

    return(list(Y = Q_learner, A = g_learner))
}

run_tmle_shift <- function(df, node_list, learner_list, tmle_spec) {
    # Note: This part usually takes a while to run
    # and if delta is too high, a large percent of the predictions get trimmed
    tmle_fit <- tmle3(tmle_spec, df, node_list, learner_list)
    return(tmle_fit)
}

resample <- function(df) {
    return(df[sample(seq_len(nrow(df)), size = nrow(df), replace = TRUE), ])
}

bootstrap_tmle_shift <- function(df, node_list, learner_list, tmle_spec, n) {
    # Get the base result from the canonical data
    base_fit <- run_tmle_shift(df, node_list, learner_list, tmle_spec)
    base_est <- base_fit$summary$psi

    # Store bootstrap estimates
    boot_vec <- numeric(n)

    for (b in seq_len(n)) {
        boot_df <- resample(df)
        boot_fit <- run_tmle_shift(boot_df, node_list, learner_list, tmle_spec)
        boot_vec[b] <- boot_fit$summary$psi
        if (b %% 1 == 0) {
            cat("Bootstraps completed:", b, "\n")
        }
    }

    # Compute CIs from bootstrap results
    est_mean <- mean(boot_vec, na.rm = TRUE)
    est_lower <- quantile(boot_vec, 0.025, na.rm = TRUE)
    est_upper <- quantile(boot_vec, 0.975, na.rm = TRUE)

    # Return df with CIs
    return(data.frame(
        est_mean = est_mean,
        est_lower = est_lower,
        est_upper = est_upper,
        est_base = base_est
    ))
}

# 9k dataset
df_9k <- read_csv(here("Outputs", "normalized_anomaly_9k.csv"))
node_list_9k <- list(
    W = c(
        "eccentricity", "obliquity", "perihelion", "insolation",
        "global_insolation", "solar_modulation", "volcanic_forcing"
    ),
    A = "co2_ppm",
    Y = "anomaly"
)
tmle_spec <- tmle_shift(
    shift_val = 0.05,
    shift_fxn = shift_additive,
    shift_fxn_inv = shift_additive_inv
)
learner_list_9k <- make_learners("9k")

set.seed(42)
bootstrap_tmle_df_9k <- bootstrap_tmle_shift(df_9k, node_list_9k, learner_list_9k, tmle_spec, n = 5)
bootstrap_tmle_df_9k
write_csv(bootstrap_tmle_df_9k, here("Outputs", "bootstrap_tmle_results_9k.csv"))

# 742k dataset
df_742k <- read_csv(here("Outputs", "normalized_anomaly_742k.csv"))
node_list_742k <- list(
    W = c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation"),
    A = "co2_ppm",
    Y = "anomaly"
)
learner_list_742k <- make_learners("742k")

set.seed(42)
bootstrap_tmle_df_742k <- bootstrap_tmle_shift(df_742k, node_list_742k, learner_list_742k, tmle_spec, n = 5)
bootstrap_tmle_df_742k
write_csv(bootstrap_tmle_df_742k, here("Outputs", "bootstrap_tmle_results_742k.csv"))
