# Execute in VS Code with `Run Source With Echo` (default `ctrl + shift + enter`)

if (!requireNamespace("here", quietly = TRUE)) {
    install.packages("here")
}
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse")
}

library(here)
library(tidyverse)

df <- read_csv(here("Outputs", "normalized_anomaly_9k.csv"))
df

# Only include treatment and confounders that need to be adjusted for
model <- lm(anomaly ~ co2_radiative_forcing + eccentricity + obliquity + perihelion + insolation + global_insolation + solar_modulation, data = df)
summary(model)
