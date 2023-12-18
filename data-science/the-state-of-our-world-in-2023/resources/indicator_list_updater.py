import pandas as pd
import openpyxl
from pathlib import Path

# Get the absolute path to the directory where this script is located
script_directory = Path(__file__).parent

# Define the path to the Excel file and the output CSV file
excel_file_path = script_directory / "resources" / "Indicator_List.xlsx"
csv_file_path = script_directory / "Indicator_List.csv"

# Load the Excel file
df = pd.read_excel(excel_file_path, sheet_name="Indicators")

# Get the relevant columns
target_cols = ["Metric Code", "Segment", "Subject", "Metric", "Source"]
df = df[target_cols]

# Write to the CSV file
df.to_csv(csv_file_path, index=False)
