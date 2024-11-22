import pandas as pd
import os

# Function to analyze each CSV file and save the results in a .txt file
def analyze_csv(file_path, output_txt_file, output_sample_file):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    with open(output_txt_file, 'w') as f:
        # Write the file name
        f.write(f"Analysis of file: {file_path}\n")
        f.write("="*50 + "\n\n")
        
        # Write the number of rows and columns
        num_rows, num_cols = df.shape
        f.write(f"Number of rows: {num_rows}\n")
        f.write(f"Number of columns: {num_cols}\n\n")
        
        # Write column names and their data types
        f.write("Columns and data types:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")
        
        # Count null values per column
        f.write("Number of null values per column:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\n")
        
        f.write("="*50 + "\n\n")
        f.write("Analysis complete.\n")
    
    # Save the first 3 rows sorted by 'job_id' to a CSV file
    if 'job_id' in df.columns:
        sample_df = df.sort_values(by='job_id').head(3)
    else:
        # If 'job_id' doesn't exist, save the first 3 rows as is
        sample_df = df.head(3)
        
    sample_df.to_csv(output_sample_file, index=False)

# Create 'outputs' directory if it doesn't exist
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of CSV files to analyze
csv_files = [
    'dataset_preprocessed_list.csv'
]

# Analyze each file and save results in the 'outputs' directory
for file in csv_files:
    # Generate file names for the analysis and sample
    output_txt_file_path = os.path.join(output_dir, f'{os.path.basename(file)}_analysis.txt')
    output_sample_file_path = os.path.join(output_dir, f'{os.path.basename(file)}_sample.csv')
    
    # Perform analysis and save the results
    analyze_csv(file, output_txt_file_path, output_sample_file_path)

print("Analysis completed and samples saved in the 'outputs' folder.")
