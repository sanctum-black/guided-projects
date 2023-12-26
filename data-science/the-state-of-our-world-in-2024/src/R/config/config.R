# BOF

# ------------------------------------------------------------------------------
# Created on Sat Nov 25 18:46:10 2023
# @author: Pablo Aguirre
# GitHub: https://github.com/pabloagn
# Website: https://pabloagn.com
# Contact: https://pabloagn.com/contact
# Part of Blog Article: the-state-of-our-world-in-2024-pt-1
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Config
# This file contains several variables &b configuration used by all other scripts.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Define Global Variables & Parameters
# ------------------------------------------------------------------------------
# Directories
modulesDir <- "src/R/modules"
rDir <- "data/raw"
utilDir <- file.path(rDir, "Utilities")
w_ResDir <- "outputs/results"
w_FigDir <- "outputs/figures"

# Plotting
color_black <- "#1a1a1a"
color_white <- "#f2f2f2"
color_gray <- "gray"

# Set up scheme as Viridis, with n colors
color_scheme <- viridis::viridis(3)

# ------------------------------------------------------------------------------
# Set Up Visualizations Config
# ------------------------------------------------------------------------------

# Set up visualizations theme
theme_set(theme_gray(base_size = 14) +
  theme(
    text = element_text(family = "Work Sans"),
    axis.text = element_text(color = color_black),
    plot.title = element_text(face = "plain", hjust = 0.5),
    panel.background = element_rect(fill = "gray90"),
    panel.grid.major = element_line(color = "white"),
    panel.grid.minor = element_line(color = "white")
  ))

# ------------------------------------------------------------------------------
# Set Up Datasets Config
# ------------------------------------------------------------------------------

# Base Columns
# ------------------------------------------------------------------------------

base_worldbank <- c(
  "country_code",
  "country_name",
  "indicator_code",
  "indicator_name"
)
