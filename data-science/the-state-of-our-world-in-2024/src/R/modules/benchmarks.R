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
# Benchmarks
# This file contains functions for running simple benchmarks for given metrics.
# ------------------------------------------------------------------------------

# Top - Bottom Entries by Column
get_top_bottom <- function(indicator, df, placement, year, n) {
  ## This function calculates top or bottom entries based on a selected metric

  # Filter data.frame
  df <- df %>%
    filter(indicator_code == indicator)

  year <- as.character(year)

  # Get top or bottom
  if (placement == "T") {
    df <- df %>%
      arrange(desc(get(year))) %>%
      slice(1:n)
  } else if (placement == "B") {
    df <- df %>%
      arrange(get(year)) %>%
      slice(1:n)
  }
  return(df)
}

# Change vs. Previous Year
get_perc_chg <- function(df, year_1, year_2) {
  return()
}


# Top - Bottom Entries by Change vs. Previous Year
get_top_bottom_pp_chg <- function() {
  return()
}


# Five-Year Change
get_fy_chg <- function() {
  return()
}


# Top - Bottom Entries by 5-Year Change
get_top_bottom_fy_chg <- function() {
  return()
}

# Average of n years
get_avg_per <- function(df, indicator, base, fy, ly) {
  ## This function calculates the average of a given metric over a period of time.
  years <- as.character((seq(fy, ly)))
  df <- df %>%
    filter(indicator_code == indicator) %>%
    rowwise() %>%
    mutate(
      mean = mean(c_across(all_of(years)), na.rm = TRUE),
      sd = sd(c_across(all_of(years)), na.rm = TRUE),
      sd_neg = sd(c_across(all_of(years))[c_across(all_of(years)) < 0], na.rm = TRUE),
      sd_pos = sd(c_across(all_of(years))[c_across(all_of(years)) > 0], na.rm = TRUE),
    )
  return(df)
}

# Top - Bottom Entries average of n years
get_top_bottom_avg_per <- function() {
  return()
}

# Ranking Stability
get_ranking_stability <- function() {
  ## This function calculates a stability score based on percent changes over time.
  return()
}

# Gap vs Global Leader
get_gap_vs_leader <- function() {
  ## This function will calculate percent gap vs global leader of a given metric.
  return()
}

# EOF
