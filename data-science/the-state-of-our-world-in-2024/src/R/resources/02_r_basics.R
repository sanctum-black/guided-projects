# ------------------------------------------------------------------------------
# Created on Sat Nov 25 18:46:10 2023
# @author: Pablo Aguirre
# GitHub: https://github.com/pabloagn
# Website: https://pabloagn.com
# Contact: https://pabloagn.com/contact
# Part of Blog Article: the-state-of-our-world-in-2024-pt-1
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# R Basics
# ------------------------------------------------------------------------------

# Load libraries
library(tidyverse)
library(readxl)
library(data.table)
library(ggalt)
library(RColorBrewer)
library(viridis)
library(car)
library(lattice)

# Define directories
rDir <- "data/raw/"
wDir <- "outputs"

# Define global params
min_year <- 1990
max_year <- 2022

# Defining custom functions
# ------------------------------------------------------------------------------
# Declare a simple function with no arguments and a return statement
myfun_1 <- function() {
  myvar_1 <- 1
  myvar_2 <- 2
  return(myvar_1 + myvar_2)
}

# Call function
myfun_1()

# Declare a simple function with arguments and a return statement
myfun_2 <- function(x, y, z) {
  return(x * y * z)
}

# Call function
myfun_2(2, 3, 4)

# Declare a function accepting another function as argument, and a return statement
inside_fun <- function(x, y) {
  return(x * y)
}

outside_fun <- function(myfun, x, y) {
  # Use inside fun
  return(myfun(x, y))
}

outside_fun(inside_fun, 2, 4)

# Data reading
# ------------------------------------------------------------------------------

# Load data using read.csv (Dataframe)
df_csv_dataframe <- read.csv(file.path(rDir,
                                       "GDP_Per_Capita",
                                       "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_6011310.csv"))

# Check object type
class(df_csv_dataframe)

# Load data using fread (data.table)
df_csv_datatable <- fread(file.path(rDir,
                                    "GDP_Per_Capita",
                                    "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_6011310.csv"),
                          header=TRUE)

# Check object type
class(df_csv_datatable)

# Load data using read_excel (Dataframe)
df_xlsx_dataframe <- read_excel(file.path(rDir,
                                          "Drug-Related_Crimes",
                                          "10.1._Drug_related_crimes.xlsx"),
                                sheet="Formal contact")

# Check object type
class(df_xlsx_dataframe)

# Transform data.frame to data.table from excel-read file
df_xlsx_datatable <- as.data.table(df_xlsx_dataframe)

# Check object type
class(df_xlsx_datatable)

# Introduction to dplyr
# ------------------------------------------------------------------------------
# Transformations using a single verb

# Get a single column
metrics_2005 <- df_csv_dataframe %>%
  select('X2005')

# Check head
head(metrics_2005)

# Filter NA
metrics_2005 <- metrics_2005 %>%
  filter(!is.na(X2005))

# Mutate using arithmetic operations
metrics_2005 <- metrics_2005 %>%
  mutate(sum_2005 = sum(X2005))

# Check head
head(metrics_2005)

# Transformations using the pipe operator (3-step)

# Perform transformation
df_2022 <- df_csv_dataframe %>%
  select(c(Country.Name,
           Country.Code,
           Indicator.Name,
           Indicator.Code,
           X2022)) %>%
  filter(!is.na(X2022)) %>%
  mutate(mean_2022 = mean(X2022))

# Get head
head(df_2022)

# Perform additional operations
unique_codes <- df_csv_dataframe %>%
  select(Country.Code) %>%
  unique() %>%
  arrange(desc(.))

# Check head
head(unique_codes)

# Get the top 10 items ordered alphabetically on descending order
unique_codes_head <- df_csv_dataframe %>%
  select(Country.Code) %>%
  unique() %>%
  arrange(desc(.)) %>%
  head(., 10)

# Check object
unique_codes_head

# Calculating aggregations
# ------------------------------------------------------------------------------
# Load built-in data
data("iris")
iris$Species %>% unique()

# Group by Species & calculate summary stats
iris_summarized <- iris %>%
  group_by(Species) %>%
  summarize(Mean.Petal.Length = mean(Petal.Length),
            Mean.Petal.Width = mean(Petal.Width))

head(iris_summarized)

# Using tidyr - pivot_longer
# ------------------------------------------------------------------------------

# Start with a dataframe that is not tidy
colnames(df_csv_dataframe)

# Tidy data (use year cols)
df_csv_dataframe_longer <- df_csv_dataframe %>%
  pivot_longer(cols = starts_with('X'),
               names_to = "Year",
               values_to = "Value",
               names_prefix = 'X')


# Check head
head(df_csv_dataframe_longer)

# Create a new df_csv_dataframe from our data.table object
# data.tables do not automatically prepend numeric col names with X
df_csv_dataframe <- as.data.frame(df_csv_datatable)

# Declare base cols (we'll respect these)
base_cols <- c("Country Name",
               "Country Code",
               "Indicator Name",
               "Indicator Code")

# User pivot_longer without column prefixes
df_csv_dataframe_longer <- df_csv_dataframe %>%
  pivot_longer(cols = !all_of(base_cols),
               names_to = "Year",
               values_to = "Values")

# Check head
head(df_csv_dataframe_longer)

# Using tidyr - pivot_wider
# ------------------------------------------------------------------------------
# Declare Dataframe
# Declare variables column 
df_variables <- c("Height",
                  "Width",
                  "X Coordinate",
                  "Y Coordinate")

# Declare values column
df_values <- c(10,
               20,
               102.3,
               102.4)

# Declare frame using vectors as columns
df_long <- data.frame(variables = df_variables,
                      values = df_values)

# Check head
head(df_long)

# Pivot wider in order to get tidy object
df_wider <- df_long %>%
  pivot_wider(names_from = variables,
              values_from = values)

# Check head
head(df_wider)

# Merging objects
# ------------------------------------------------------------------------------

# Using Base R merge()
# Define two frames
df_countries <- data.frame(country_key = c("SEN", "LTU", "ISL", "GBR"),
                           country = c("Senegal", "Lituania", "Iceland", "United Kingdom"))

df_metrics_1 <- data.frame(country_key = c("LTU", "MEX", "GBR", "SEN", "ISL"),
                         metric = c(23421, 234124, 25345, 124390, 34634))

df_metrics_2 <- df_metrics <- data.frame(country_key = c("LTU", "SEN", "CHE", "GBR", "ISL"),
                                         metric = c(37824, 89245, 28975, 49584, 29384))

# Create new frame merging left for df_countries on key
df_left_base <- merge(df_countries, df_metrics_1, by = "country_key", all.x = TRUE)

# Check head
head(df_left_base)

# Create new frame merging right for df_metrics on key
df_right_base <- merge(df_countries, df_metrics_1, by = "country_key", all.y = TRUE)

# Check head
head(df_right_base)

# Create new frame merging inner on key
df_inner_base <- merge(df_countries, df_metrics_1, by = "country_key", all = FALSE)

# Check head
head(df_inner_base)

# Create a new frame by merging df_left_base with df_metrics_2 using left_merge
df_left_base <- merge(df_left_base, df_metrics_2, by = "country_key", all.x = TRUE)

# Check head
head(df_left_base)


# Using dplyr 
df_left_dplyr <- df_countries %>%
  left_join(df_metrics_1, by = "country_key") %>%
  inner_join(df_metrics_2, by = "country_key")

# Check head
head(df_left_dplyr)


# Using purr's reduce() function
df_list <- list(df_countries,
                df_metrics_1,
                df_metrics_2)

df_left_reduce <- df_list %>%
  reduce(left_join, by = "country_key")

# Check head
head(df_left_reduce)

# Classical Statistics
# ------------------------------------------------------------------------------

# Create a set of observations
n <- nrow(df_csv_dataframe)

set.seed(1)
dataset_1 <- runif(n = n, min = 1, max = 500000)

set.seed(2)
dataset_2 <- runif(n = n, min = 1, max = 500000)

# Build frame
df_stats_1 <- df_csv_dataframe %>%
  select(`Country Code`) %>%
  mutate(metric_1 = dataset_1,
         metric_2 = dataset_2)

# Descriptive Statistics

# Calculate summary stats for specific columns
summary(df_csv_dataframe$`2022`)

# Calculate single statistics for a given column
mean(df_csv_dataframe$`2022`, na.rm = TRUE)
median(df_csv_dataframe$`2022`, na.rm = TRUE)
sqrt(var(df_csv_dataframe$`2022`, na.rm = TRUE))
sd(df_csv_dataframe$`2022`, na.rm = TRUE)
quantile(df_csv_dataframe$`2022`, na.rm = TRUE)
sum(is.na(df_csv_dataframe$`2022`))

# Probability Distributions

# Create sets of random numbers drawn from distributions
# Declare target number of observations
n <- 10

# Normal
rnorm(n = n, mean = 0, sd = 1)

# Binomial
rbinom(n = n, size = 50, prob = 0.5)

# Uniform
runif(n = n, min = 0, max = 1)

# Poisson
rpois(n = n, lambda = 5)

# Exponential
rexp(n = n, rate = 1)

# Comparing real vs theoretical distributions
# Load data
data(USArrests)

# Compare normal distribution with urban population variable
# Calculate mean & sd
mean_urbanpop <- mean(USArrests$UrbanPop)
sd_urbanpop <- sd(USArrests$UrbanPop)

# Plot a histogram
ggplot(USArrests, aes(x = UrbanPop)) +
  geom_histogram(aes(y = ..density..), binwidth = 5, fill = "lightblue", color = "black") +
  stat_function(fun = dnorm, args = list(mean = mean_urbanpop, sd = sd_urbanpop), color = "red", size = 1) +
  labs(title = "Histogram of Urban Population Percentage with Normal Distribution", 
       x = "Urban Population Percentage", 
       y = "Density") +
  theme_gray()

# Correlation and Regression Analysis
# ------------------------------------------------------------------------------
# Load data and check README
data(swiss)
?swiss

# Check dimensions & cols
dim(swiss)
names(swiss)

# Explore linear correlation between variables
cor(swiss)

# Create pairs plot
pairs(swiss, panel = panel.smooth, main = "Swiss Dataset Pair Plots")

# Plotting
# ------------------------------------------------------------------------------
# Line chart
# Recalling the head of our frame
head(df_csv_dataframe_longer)

# Reduce set by selecting a subset of countries & years
country_scope <- c("BRA",
                   "CHN",
                   "FRA")

year_scope <- as.character(
  seq(2010, 2020)
  )

# Filter dataframe
df_csv_dataframe_longer <- df_csv_dataframe_longer %>%
  filter((Year %in% year_scope) & (`Country Code` %in% country_scope))

# Check head
head(df_csv_dataframe_longer)

# Check shape
count(df_csv_dataframe_longer)

# Check data types first
str(df_csv_dataframe_longer)

# We must first convert Year to numeric
# Note: Here we use Base R syntax since it's more direct for this case.
df_csv_dataframe_longer$Year <- as.integer(df_csv_dataframe_longer$Year)

# However, we can also use dplyr syntax
#df_csv_dataframe_longer <- df_csv_dataframe_longer %>%
#  mutate(Year = integer(Year))

# Generate plot
ggplot(data = df_csv_dataframe_longer,
       mapping = aes(x = Year, y = Values, color = `Country Name`)) +
  geom_line() + 
  theme_gray() + 
  labs(title = "GDP per capita, PPP (current international $)",
       x = "Year",
       y = "Metric",
       color = "Country") + 
  scale_x_continuous(breaks = pretty(df_csv_dataframe_longer$Year, n = 5))

# Boxplot (Multiple)
# Declare figure
boxplot <- ggplot(data = iris,
                  aes(x = Species, y = Sepal.Length)) + 
  geom_boxplot(outlier.colour="black",
               outlier.shape=16,
               outlier.size=2,
               notch=TRUE)

# Print figure
boxplot

# Declare figure
jitterplot <- boxplot + 
  geom_jitter(shape=16, position=position_jitter(0.2))

# Print figure
jitterplot