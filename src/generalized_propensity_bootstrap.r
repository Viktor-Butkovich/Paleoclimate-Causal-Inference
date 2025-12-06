options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("here", quietly = TRUE)) {
    install.packages("here")
}
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse")
}

library(here)
library(tidyverse)

run_gps_dose_response <- function(df, confounders) {
    # Fit treatment model (treatment ~ confounders)
    gps_model <- lm(
        co2_radiative_forcing ~ .,
        data = df[, c("co2_radiative_forcing", confounders)]
    )

    df$mu_hat <- predict(gps_model) # Predicted mean of treatment for each unit
    sigma_hat <- sd(gps_model$residuals) # residual SD of treatment

    # Compute actual GPS values for outcome model
    df$gps_hat <- dnorm(df$co2_radiative_forcing, mean = df$mu_hat, sd = sigma_hat)

    # Outcome model: outcome ~ treatment + GPS + interaction
    # Hirano & Imbens (2004) recommends an interaction term to reduce bias
    outcome_model <- lm(anomaly ~ co2_radiative_forcing * gps_hat, data = df)

    # Dose–response estimation function
    estimate_dose_response <- function(df, A_seq, sigma_hat, outcome_model) {
        results <- numeric(length(A_seq))
        for (j in seq_along(A_seq)) {
            a_val <- A_seq[j]

            # GPS evaluated at hypothetical dose 'a' for every unit
            gps_a <- dnorm(a_val, mean = df$mu_hat, sd = sigma_hat)

            # predicted Y(a)
            pred_y <- predict(
                outcome_model,
                newdata = data.frame(
                    co2_radiative_forcing = rep(a_val, nrow(df)),
                    gps_hat = gps_a
                )
            )

            # average = E[Y(a)]
            results[j] <- mean(pred_y)
        }
        return(results)
    }

    # Define dose range
    A_seq <- seq(min(df$co2_radiative_forcing), max(df$co2_radiative_forcing), length.out = 200)

    # Compute dose–response curve
    drf <- estimate_dose_response(df, A_seq, sigma_hat, outcome_model)

    return(data.frame(co2_radiative_forcing = A_seq, drf = drf))
}

resample <- function(df) {
    return(df[sample(seq_len(nrow(df)), size = nrow(df), replace = TRUE), ])
}

bootstrap_gps_dose_response <- function(df, confounders, n) {
    # Get the base result from the canonical data
    base_result <- run_gps_dose_response(df, confounders)
    A_seq <- base_result$co2_radiative_forcing

    # Store bootstrap results in a matrix (repliace rows, dose point columns)
    boot_mat <- matrix(NA, nrow = n, ncol = length(A_seq))

    for (b in seq_len(n)) {
        boot_df <- resample(df)
        boot_result <- run_gps_dose_response(boot_df, confounders)
        boot_mat[b, ] <- boot_result$drf
        if (b %% 25 == 0) {
            cat("Bootstraps completed:", b, "\n")
        }
    }

    # Compute CIs from bootstrap results
    drf_mean <- colMeans(boot_mat, na.rm = TRUE)
    drf_lower <- apply(boot_mat, 2, quantile, probs = 0.025, na.rm = TRUE)
    drf_upper <- apply(boot_mat, 2, quantile, probs = 0.975, na.rm = TRUE)

    # Return df with CIs
    return(data.frame(
        co2_radiative_forcing = A_seq,
        drf_mean = drf_mean,
        drf_lower = drf_lower,
        drf_upper = drf_upper,
        drf_base = base_result$drf
    ))
}

plot_bootstrap <- function(bootstrap_df, name, file_path) {
    # Reshape into long format for the two curves
    lines_df <- bootstrap_df %>%
        select(co2_radiative_forcing, drf_base, drf_mean) %>%
        pivot_longer(
            cols = c(drf_base, drf_mean),
            names_to = "curve",
            values_to = "drf"
        )

    dose_response_plot <- ggplot() +
        # Confidence interval ribbon from bootstrap_df
        geom_ribbon(
            data = bootstrap_df,
            aes(
                x = co2_radiative_forcing,
                ymin = drf_lower,
                ymax = drf_upper
            ),
            fill = "skyblue", alpha = 0.3
        ) +
        # Lines from lines_df with mapped aesthetics
        geom_line(
            data = lines_df,
            aes(
                x = co2_radiative_forcing,
                y = drf,
                color = curve,
                linetype = curve
            ),
            linewidth = 1.2
        ) +
        scale_color_manual(
            values = c("drf_base" = "black", "drf_mean" = "red"),
            labels = c("Actual estimate", "Mean of bootstrap estimates")
        ) +
        scale_linetype_manual(
            values = c("drf_base" = "solid", "drf_mean" = "dashed"),
            labels = c("Actual estimate", "Mean of bootstrap estimates")
        ) +
        labs(
            x = "CO2 radiative forcing",
            y = "Estimated temperature anomaly (Degrees C)",
            title = name,
            color = "Curve",
            linetype = "Curve"
        ) +
        theme_minimal()


    ggsave(
        filename = here("Outputs", file_path),
        plot = dose_response_plot,
        width = 12,
        height = 6,
        dpi = 300,
        bg = "white"
    )
}

# 742k dataset
df_742k <- read_csv(here("Outputs", "anomaly_742k.csv"))
confounders <- c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation")
set.seed(42)
bootstrap_df_742k <- bootstrap_gps_dose_response(df_742k, confounders, n = 1000)
head(bootstrap_df_742k)
plot_bootstrap(
    bootstrap_df_742k,
    "Dose-Response Function (with 95% Bootstrap Confidence Interval) from Past 742k Years",
    "GPS_Dose_Response_Function_Bootstrap_742k.png"
)

# 9k dataset
df_9k <- read_csv(here("Outputs", "anomaly_9k.csv"))
confounders <- c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation", "solar_modulation", "volcanic_forcing")
set.seed(42)
bootstrap_df_9k <- bootstrap_gps_dose_response(df_9k, confounders, n = 1000)
head(bootstrap_df_9k)
plot_bootstrap(
    bootstrap_df_9k,
    "Dose-Response Function (with 95% Bootstrap Confidence Interval) from Past 9k Years",
    "GPS_Dose_Response_Function_Bootstrap_9k.png"
)
