if (!requireNamespace("dagitty", quietly = TRUE)) {
    install.packages("dagitty")
}
if (!requireNamespace("ggdag", quietly = TRUE)) {
    install.packages("ggdag")
}
if (!requireNamespace("here", quietly = TRUE)) {
    install.packages("here")
}
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse")
}

library(dagitty)
library(ggdag)
library(here)
library(tidyverse)

# Annotations
annotations <- c(
    A = "A: CO2 concentration",
    X1 = "X1: Milankovitch cycles",
    X2 = "X2: Beryllium-10 (cosmic ray flux proxy, unused)",
    X3 = "X3: VADM (geomagnetic proxy, unused)",
    X4 = "X4: Solar modulation",
    X5 = "X5: CO2 radiative forcing",
    X6 = "X6: Volcanic forcing",
    U1 = "U1: Actual anomaly",
    U2 = "U2: Magnetic field strength",
    U3 = "U3: Cosmic ray flux",
    Y = "Y: Measured anomaly (via proxies)",
    C = "Measured climate drivers: Volcanic forcing,\n solar modulation, milankovitch cycles",
    U = "Unmeasured climate drivers: Magnetic field\n strength, Earth's geological processes",
    M = "Climate mediators: Soil/rock processes, vegetation cover,\n ocean circulation, ice sheet dynamics"
)

unmeasured_color <- "#E8E8E8" # Light gray for unmeasured
variable_color <- "#4A90E2" # Blue for measured covariates
treatment_color <- "#E85D5D" # Red for treatment
outcome_color <- "#5CB85C" # Green for outcome

# Map nodes to their types
node_types <- c(
    A = "Treatment",
    X1 = "Measured",
    X2 = "Measured",
    X3 = "Measured",
    X4 = "Measured",
    X5 = "Measured",
    X6 = "Measured",
    U1 = "Unmeasured",
    U2 = "Unmeasured",
    U3 = "Unmeasured",
    Y = "Outcome",
    C = "Measured",
    U = "Unmeasured",
    M = "Unmeasured"
)

# Map types to colors
type_colors <- c(
    Treatment = treatment_color,
    Measured = variable_color,
    Unmeasured = unmeasured_color,
    Outcome = outcome_color
)

# Define DAG
g <- dagitty("dag {
  A -> X5
  X1 -> U1
  X1 -> A
  X4 -> U3
  X5 -> U1
  X6 -> U1
  X6 -> A
  U1 -> Y
  U2 -> X3
  U2 -> U3
  U3 -> X2
  U3 -> U1
  U3 -> A
  C -> M
  U -> M
  M -> U1
  M -> A
}")

# Manual coordinate assignment
coords <- list(
    x = c(
        X1 = 0.9, X4 = 1.5, U2 = 1.8,
        A = 1.3, U3 = 1.5, X3 = 1.8,
        X5 = 1, X6 = 1.0, X2 = 1.65,
        U1 = 0.9,
        Y = 0.9, C = 2.0, U = 2.4, M = 2.2
    ),
    y = c(
        X1 = 6, X4 = 6, U2 = 6,
        A = 4, U3 = 5, X3 = 5,
        X5 = 3.3, X6 = 4.7, X2 = 4.2,
        U1 = 2,
        Y = 1, C = 4, U = 4, M = 3
    )
)

# Convert to tidy df and assign coordinates
coord_df <- coords2df(coords)
coordinates(g) <- coords2list(coord_df)

ggdag(g, text = FALSE) +
    geom_dag_edges() +
    geom_dag_point(aes(fill = node_types[name]), shape = 21, color = "black") +
    geom_label(aes(x = x, y = y - 0.25, label = annotations[name]),
        fill = "white",
        color = "black",
        size = 4,
        label.padding = unit(0.15, "lines"),
        label.size = 0
    ) +
    scale_fill_manual(values = type_colors, name = "Variable Type") +
    theme_dag_gray() +
    ggtitle("Paleoclimate Causal DAG") +
    theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "right"
    )
ggsave(
    filename = here("Outputs", "Paleoclimate_Causal_DAG.png"),
    width = 18,
    height = 12,
    dpi = 300
)
