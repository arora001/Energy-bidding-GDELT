import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Construct the full path to the CSV file
csv_path = os.path.join(script_dir, 'weather_energy_15min.csv')

# Check if the file exists before attempting to read it
if os.path.exists(csv_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Check if the 'Power demand_sum' column exists
    if 'Power demand_sum' in df.columns:
        # Calculate the summary statistics for the 'Power demand_sum' column
        energy_summary = df['Power demand_sum'].describe()

        # Print the summary
        print("Summary of the 'Power demand_sum' field:")
        print(energy_summary)
    else:
        print(f"Error: 'Power demand_sum' column not found in {csv_path}")
else:
    print(f"Error: File not found at {csv_path}")
