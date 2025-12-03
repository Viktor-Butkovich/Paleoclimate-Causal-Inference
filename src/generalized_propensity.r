if (!requireNamespace("here", quietly = TRUE)) {
    install.packages("here")
}
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse")
}

library(here)
library(tidyverse)

df <- read_csv(here("Outputs", "anomaly_742k.csv"))
df

confounders <- c("eccentricity", "obliquity", "perihelion", "insolation", "global_insolation") #anomaly_724k.csv does not have solar_modulation

# Fit treatment model (treatment ~ confounders)
gps_model <- lm(co2_radiative_forcing ~ eccentricity + obliquity + perihelion + insolation + global_insolation, data = df)

df$mu_hat <- predict(gps_model) # Predicted mean of treatment for each unit
sigma_hat <- sd(gps_model$residuals) # residual SD of treatment

# Compute actual GPS values for outcome model
df$gps_hat <- dnorm(df$co2_radiative_forcing, mean = df$mu_hat, sd = sigma_hat)

# Check balance of GPS strata
df_balance <- df %>%
    mutate(gps_strata = ntile(gps_hat, 10))  # 10 equal-sized strata

# Compute mean of each confounder in each strata
balance_summary <- df_balance %>%
    group_by(gps_strata) %>%
    summarize(
        across(
            confounders,
            ~ mean(.x, na.rm = TRUE),
            .names = "mean_{col}"
        )
    )

print(balance_summary)

# Plot each confounder's mean vs. stratum
for (cov in confounders) {
    p <- df_balance %>%
      group_by(gps_strata) %>%
      summarize(mean_val = mean(.data[[cov]])) %>%
      ggplot(aes(x = gps_strata, y = mean_val)) +
      geom_line() +
      labs(title = paste("Balance: ", cov))
    
    ggsave(
      filename = here("Outputs", paste0("GPS_Balance_", cov, ".png")),
      plot = p,
      width = 12, height = 6, dpi = 300, bg = "white"
    )
}

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

plot_df <- data.frame(
    co2_radiative_forcing = A_seq,
    drf = drf
)

# Plot dose–response curve
dose_response_plot <- ggplot(plot_df, aes(x = co2_radiative_forcing, y = drf)) +
    geom_line(linewidth = 1.3) +
    labs(
        x = "CO₂ radiative forcing",
        y = "Estimated temperature anomaly",
        title = "Dose–Response Function"
    )
ggsave(
    filename = here("Outputs", "GPS_Dose_Response_Function.png"),
    plot = dose_response_plot,
    width = 12,
    height = 6,
    dpi = 300,
    bg = "white"
)