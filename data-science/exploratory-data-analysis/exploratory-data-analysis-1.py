"""
Created on Fri Feb 03 19:23:00 2023
@author: Pablo Aguirre
GitHub: https://github.com/pabloagn
Website: https://pabloagn.com
Contact: https://pabloagn.com/contact
Part of Guided Project: exploratory-data-analysis
"""

"""
Problem Statement
-----------------
Our client, an insurance company, has asked us to conduct a correlational study
between different behavioral & genetic characteristics and lung cancer incidence.
For this, they have provided a medical dataset containing a set of anonymous patients
along with their medical file generated upon hospital admission.
"""

"""
Data Set
-----------------
https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
"""

# Understanding our data set
# ---------------------------

# Import required modules
# Data manipulation modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter

# System utility modules
import os
import shutil
from pathlib import Path

# Plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Models
from scipy.stats import spearmanr

# Defining plot parameters

# Before anything else, delete the Matplotlib
# font cache directory if it exists, to ensure
# custom font propper loading
try:
    shutil.rmtree(matplotlib.get_cachedir())
except FileNotFoundError:
    pass

# Define main color as hex
color_main = '#1a1a1a'

# Define title & label padding
text_padding = 18

# Define font sizes
title_font_size = 17
label_font_size = 14

# Define rc params
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Lora']

# Read the data set
df = pd.read_csv('cancer patient data sets.csv', index_col=None)

# Data set shape and types
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.isnull().values.any())

# Remove index column
df.drop(columns = "index", inplace = True)

# Number of unique patients
print(df['Patient Id'].nunique())

# Understanding our data set
# ---------------------------

# Overal description
round(df.describe().iloc[1:, ].T, 2)

# Age range & description
print(sorted(list(df['Age'].unique())))

# Age histogram
# Create figure
plt.figure('Patient Age Distribution')

# Plot the age distribution
sns.kdeplot(df['Age'], color = color_main)

# Enable grid
plt.grid(True, zorder=0)

# Set xlabel and ylabel
plt.xlabel("Age", fontsize=label_font_size, labelpad=text_padding)
plt.ylabel("Patient Density", fontsize=label_font_size, labelpad=text_padding)

# Remove bottom and top separators
sns.despine(bottom=True)

# Add plot title
plt.title('Patient Age KDE', fontsize=title_font_size, pad=text_padding)

# Optional: Save the figure as a png image
plt.savefig('plots/' + '01_patient_age_KDE.png', format = 'png', dpi = 300, transparent = True)

# Close the figure
plt.close()


# Age vs illness severity boxplot
# Create figure
plt.figure('Patient Age Distribution For Different Severities')

# Plot the age distribution for different illness severities
sns.boxplot(x='Level',
            y='Age',
            data=df,
            order=["Low", "Medium", "High"],
            color=color_main,
            medianprops=dict(color="white", label='median'),
            boxprops=dict(alpha=0.8))

# Enable grid
plt.grid(True, zorder=0)

# Set xlabel and ylabel
plt.xlabel("Illness Severity", fontsize=label_font_size, labelpad=text_padding)
plt.ylabel("Patient Age Distribution", fontsize=label_font_size, labelpad=text_padding)

# Remove bottom and top separators
sns.despine(bottom=True)

# Add plot title
plt.title('Patient Age Distribution For Different Severities', fontsize=title_font_size, pad=text_padding)

# Remove subplot title
plt.suptitle('')

# Optional: Save the figure as a png image
plt.savefig('plots/' + '02_patient_age_distribution_for_different_severities.png', format = 'png', dpi = 300, transparent = True)

# Close the figure
plt.close()


# Gender vs illness severity bar chart
# Create figure
plt.figure('Patient Gender Composition For Different Severities')

# Plot the age distribution for different illness severities
df_group = df.groupby(['Level', 'Gender'])['Patient Id'].count().reset_index()
sns.catplot(data=df_group,
            kind="bar",
            x="Level",
            y="Patient Id",
            hue="Gender",
            palette = sns.color_palette("rocket"),
            alpha=0.8,
            order=["Low", "Medium", "High"]
            )

# Enable grid
plt.grid(True, zorder=0)

# Set xlabel and ylabel
plt.xlabel("Illness Severity", fontsize=label_font_size, labelpad=text_padding)
plt.ylabel("Number of Patients", fontsize=label_font_size, labelpad=text_padding)

# Remove bottom and top separators
sns.despine(bottom=True)

# Add plot title
plt.title('Number of Patients For Different Severities', fontsize=title_font_size, pad=text_padding)

# Remove subplot title
plt.suptitle('')

# Optional: Save the figure as a png image
plt.savefig('plots/' + '03_patient_gender_count_for_different_severities.png', format = 'png', dpi = 300, transparent = True)

# Close the figure
plt.close()


# Monotonicity Verification
# Define variables to keep
ordinal_vars = ['Patient Id',
                'Age',
                'Air Pollution',
                'Alcohol use',
                'Dust Allergy',
                'OccuPational Hazards',
                'Genetic Risk',
                'chronic Lung Disease',
                'Balanced Diet',
                'Obesity',
                'Smoking',
                'Passive Smoker',
                'Level']

# Create new Data Frame
df_corr = df[ordinal_vars]

# Map Level to numeric values
illness_level_dict = {'Low' : 1,
                      'Medium' : 2,
                      'High': 3}

df_corr['Level'] = df_corr['Level'].map(illness_level_dict)

# Remove Patient Id variable for correlation study
df_corr_m = df_corr.drop(columns = ['Patient Id'])

# Plot a pair plot
# Create figure
plt.figure('Pair Plot', figsize=(20, 20))

# Plot Pair Plot
g = sns.pairplot(df_corr_m)

# Enable grid
plt.grid(True, zorder=0)

# Remove bottom and top separators
sns.despine(bottom=True)

# Add plot title
g.fig.suptitle('Pairplot for Risk Factor Variables', y=1.02, fontsize=title_font_size)

# Optional: Save the figure as a png image
plt.savefig('plots/' + '04_pairplot_risk_factor_categorical_variables.png', format = 'png', dpi = 300, transparent = True)

# Close the figure
plt.close()


# Spearman Correlation Analysis
# Create figure
plt.figure('Spearman Correlation Heatmap for Risk Factor Variables', figsize=(20,18))

# Create the correlation matrix
df_corr_ms = df_corr_m.corr(method='spearman')

# Plot using heat man
sns.heatmap(round(df_corr_ms, 2), annot=True, cmap=sns.cm.rocket_r)

# Enable grid
plt.grid(True, zorder=0)

# Remove bottom and top separators
sns.despine(bottom=True)

# Add plot title
plt.title('Spearman Correlation Heatmap for Risk Factor Variables', fontsize=title_font_size, pad=text_padding)

# Remove subplot title
plt.suptitle('')

# Optional: Save the figure as a png image
plt.savefig('plots/' + '05_spearman_correlation_heatmap_risk_factor_categorical_variables.png', format = 'png', dpi = 300, transparent = True)

# Close the figure
plt.close()

# p-value Test
# Copy of dataframe to compare against itself
df_corr_c = df_corr_m.copy()
pvalmat = np.zeros((df_corr_m.shape[1], df_corr_c.shape[1]))

for i in range(df_corr_m.shape[1]):    
    for j in range(df_corr_c.shape[1]):
        # Pearson correlation test        
        corrtest = spearmanr(df_corr_m[df_corr_m.columns[i]], df_corr_c[df_corr_c.columns[j]])  
        pvalmat[i,j] = corrtest[1]
        
# Dataframe for p-values
df_pvals_ms = pd.DataFrame(pvalmat, columns=df_corr_m.columns, index=df_corr_c.columns)

# Round results
df_corr_ms = round(df_corr_ms, 4)
df_pvals_ms = round(df_pvals_ms, 6)

# Export to Excel sheet
writer = pd.ExcelWriter('outputs/Risk_Factor_Analysis.xlsx', engine = 'xlsxwriter')
df_corr_ms.to_excel(writer, sheet_name = 'Spearman_Corr_Coef')
df_pvals_ms.to_excel(writer, sheet_name = 'Spearman_Pvals')
writer.close()

# Export to csv for use in GitHub Gists
df_corr_ms.to_csv('outputs/Spearman_Corr_Coef.csv', sep = ',')
df_pvals_ms.to_csv('outputs/Spearman_Pvals.csv', sep = ',')