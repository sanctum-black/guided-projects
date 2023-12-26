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
# 1. Macroeconomics
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
library(readr)
library(readxl)
library(writexl)
library(openxlsx)
library(dplyr)
library(tidyr)
library(data.table)
library(stringr)
library(lubridate)
library(car)
library(broom)
library(ggplot2)
library(ggalt)
library(RColorBrewer)
library(extrafont)
library(viridis)

# ------------------------------------------------------------------------------
# Source scripts
# ------------------------------------------------------------------------------
# Config
configDir <- "src/R/config"
source(file.path(configDir, "config.R"))

# Modules
source(file.path(modulesDir, "benchmarks.R"))
source(file.path(modulesDir, "preprocessing.R"))
source(file.path(modulesDir, "statistics.R"))

# ------------------------------------------------------------------------------
# GDP
# ------------------------------------------------------------------------------

# Includes the following indexes:
# - GDP Nominal
# - GDP PPP
# - GDP Per Capita PPP
# - GDP Growth Rate
# - GDP Per Capita Growth Rate

# Define read functions
read_data_csv <- function(folder, path) {
  # Read data
  df <- read_csv(file.path(rDir, folder, path), show_col_types = FALSE)
  return(df)
}

# Get countries & prefilters
get_utils <- function() {
  # Get country data.frame
  df_countries <- read_data_csv("Utilities", "Countries_Baseline.csv") %>%
    rename("country_code" = "iso3166_1_alpha_3")

  # Get country codes list
  prefilter_countries <- pull(df_countries, country_code)

  # Return vector of results
  return(list(df_countries, prefilter_countries))
}

# Define merge countries function
merge_countries <- function(df, df_countries, mode, world) {
  # If mode is simple (S), get only the country codes from the df_countries frame
  if (mode == "S") {
    # Filter by columns
    df_countries <- df_countries %>%
      select(any_of(c("country_code", "country_name")))
    # If world data is required
    if (world) {
      df_world <- data.frame(
        "country_code" = "WLD",
        "country_name" = "World"
      )
      df_countries <- rbind(df_countries, df_world)
    }
    # Merge with df_countries
    df <- merge(df_countries, df, by = "country_code", all.x = TRUE)
  } else {
    df <- df
  }
  return(df)
}

# Define preprocess functions
preprocess_worldbank <- function(read_fun, folder_dir, file_dir, min_year, max_year, mode = "S", world = FALSE) {
  # Get data
  df <- read_fun(folder_dir, file_dir)

  # Rename columns
  df <- df %>%
    rename(
      country_code = "Country Code",
      indicator_name = "Indicator Name",
      indicator_code = "Indicator Code"
    )

  # Define target cols
  target_cols <- c(
    "country_code",
    "indicator_name",
    "indicator_code",
    seq(
      min_year,
      max_year,
      1
    )
  )

  # Filter columns
  df <- df %>% select(any_of(target_cols))

  # Merge with df_countries
  df <- merge_countries(df, df_countries, "S", world)

  # Return filtered df
  return(df)
}

# Get util objects
utils_objects <- get_utils()
df_countries <- utils_objects[[1]]
prefilter_countries <- utils_objects[[2]]

# Define a transformation function
# This will be used for dealing with visualizations

# Define parameters for GDP metrics
gdp_min_year <- 1990
gdp_max_year <- 2022

# Load data
df_gdp_nom <- preprocess_worldbank(read_data_csv,
  "GDP_Nominal",
  "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6011335.csv",
  gdp_min_year,
  gdp_max_year,
  world = TRUE
)

df_gdp_ppp <- preprocess_worldbank(read_data_csv,
  "GDP_PPP",
  "API_NY.GDP.MKTP.PP.CD_DS2_en_csv_v2_5996066.csv",
  gdp_min_year,
  gdp_max_year,
  world = TRUE
)

df_gdp_pc <- preprocess_worldbank(read_data_csv,
  "GDP_Per_Capita",
  "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_6011310.csv",
  gdp_min_year,
  gdp_max_year,
  world = TRUE
)

df_gdp_gr <- preprocess_worldbank(read_data_csv,
  "GDP_Growth",
  "API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_5994650.csv",
  gdp_min_year,
  gdp_max_year,
  world = TRUE
)

df_gdp_pc_gr <- preprocess_worldbank(read_data_csv,
  "GDP_Per_Capita_Growth",
  "API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5994795.csv",
  gdp_min_year,
  gdp_max_year,
  world = TRUE
)
# Define year under study
target_year <- 2022
target_year <- as.character(target_year)

# GDP - Nominal
# ------------------------------------------------------------------------------
# Filter top & bottom 10 items
df_gdp_nom_top <- get_top_bottom(
  indicator = "NY.GDP.MKTP.CD",
  df = df_gdp_nom,
  placement = "T",
  year = target_year,
  n = 10
)

# Calculate percentage  of top 10 economies vs the world
gdp_nom_world_2022 <- df_gdp_nom_top %>%
  filter(country_code == "WLD") %>%
  pull(target_year) %>%
  first()

df_gdp_nom_top <- df_gdp_nom_top %>%
  mutate(world_2022 = gdp_nom_world_2022) %>%
  mutate(perc_2022 = get(target_year) / world_2022)

# GDP - PPP
# ------------------------------------------------------------------------------
# Filter top & bottom 10 items
df_gdp_ppp_top <- get_top_bottom(
  indicator = "NY.GDP.MKTP.PP.CD",
  df = df_gdp_ppp,
  placement = "T",
  year = target_year,
  n = 10
)

# Calculate percentage  of top 10 economies vs the world
gdp_ppp_world_2022 <- df_gdp_ppp_top %>%
  filter(country_code == "WLD") %>%
  pull(`2022`) %>%
  first()

df_gdp_ppp_top <- df_gdp_ppp_top %>%
  mutate(world_2022 = gdp_ppp_world_2022) %>%
  mutate(perc_2022 = `2022` / world_2022)

# GDP - Per Capita PPP
# ------------------------------------------------------------------------------
# Filter top & bottom 10 & 30 items
df_gdp_pc_top_10 <- get_top_bottom(
  indicator = "NY.GDP.PCAP.PP.CD",
  df = df_gdp_pc,
  placement = "T",
  year = target_year,
  n = 10
)

df_gdp_pc_bot_10 <- get_top_bottom(
  indicator = "NY.GDP.PCAP.PP.CD",
  df = df_gdp_pc,
  placement = "B",
  year = target_year,
  n = 10
)

df_gdp_pc_top_30 <- get_top_bottom(
  indicator = "NY.GDP.PCAP.PP.CD",
  df = df_gdp_pc,
  placement = "T",
  year = target_year,
  n = 30
)

df_gdp_pc_bot_30 <- get_top_bottom(
  indicator = "NY.GDP.PCAP.PP.CD",
  df = df_gdp_pc,
  placement = "B",
  year = target_year,
  n = 30
)

# Add world to both
df_gdp_pc_world <- df_gdp_pc %>%
  filter(country_code == "WLD")

df_gdp_pc_top_10 <- bind_rows(df_gdp_pc_top_10, df_gdp_pc_world)
df_gdp_pc_bot_10 <- bind_rows(df_gdp_pc_bot_10, df_gdp_pc_world)

df_gdp_pc_top_30 <- bind_rows(df_gdp_pc_top_30, df_gdp_pc_world)
df_gdp_pc_bot_30 <- bind_rows(df_gdp_pc_bot_30, df_gdp_pc_world)

# Where do top performing Nominal & PPP Economies go vs GDP PC PPP
df_gdp_pc_sorted <- df_gdp_pc %>%
  arrange(desc(get(target_year))) %>%
  mutate(rank = min_rank(desc(`2022`)))

# GDP - Per Capita PPP
# ------------------------------------------------------------------------------

# Calculate average over last n years
df_gdp_pc_gr_avg <- get_avg_per(
  df_gdp_pc_gr,
  "NY.GDP.PCAP.KD.ZG",
  base_worldbank,
  2012,
  2022
)

# Next, add GDP per Capita for each country (last year)
df_gdp_pc_single <- df_gdp_pc %>%
  select(c("country_code", "2022")) %>%
  rename("GDP_PC_PPP_2022" = "2022")

df_gdp_pc_gr_avg <- df_gdp_pc_gr_avg %>%
  merge(df_gdp_pc_single, by = "country_code", all = TRUE)









# EOF
