import pandas as pd
import openpyxl
import os

# Load Excel file
df = pd.read_excel("Indicator_List.xlsx", sheet_name="Indicators")

# Get cols
target_cols = ["Metric Code", "Segment", "Subject", "Metric", "Source"]
df = df[target_cols]

# Write
df.to_csv("Indicator_List.csv", index=False)
