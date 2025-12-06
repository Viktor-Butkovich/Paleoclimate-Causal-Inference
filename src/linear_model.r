# Execute in VS Code with `Run Source With Echo` (default `ctrl + shift + enter`)
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("here", quietly = TRUE)) {
    install.packages("here")
}
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse")
}
if (!requireNamespace("broom", quietly = TRUE)) {
    install.packages("broom")
}

library(here)
library(tidyverse)
library(broom)

df_9k <- read_csv(here("Outputs", "normalized_anomaly_9k.csv"))
df_9k

# Only include treatment and confounders that need to be adjusted for
model_9k <- lm(anomaly ~ co2_radiative_forcing + eccentricity + obliquity + perihelion + insolation + global_insolation + solar_modulation + volcanic_forcing, data = df_9k)
summary(model_9k)
tidy_model_9k <- broom::tidy(model_9k, conf.int = TRUE)
write_csv(tidy_model_9k, here("Outputs", "linear_model_9k_results.csv"))

df_742k <- read_csv(here("Outputs", "normalized_anomaly_742k.csv"))
df_742k

model_742k <- lm(anomaly ~ co2_radiative_forcing + eccentricity + obliquity + perihelion + insolation + global_insolation, data = df_742k)
summary(model_742k)
tidy_model_742k <- broom::tidy(model_742k, conf.int = TRUE)
write_csv(tidy_model_742k, here("Outputs", "linear_model_742k_results.csv"))
