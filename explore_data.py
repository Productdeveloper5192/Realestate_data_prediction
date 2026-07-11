import pandas as pd
import numpy as np

# Load data
user_df = pd.read_excel('Case Study 2 Data.xlsx', sheet_name='User Data')
prop_df = pd.read_excel('Case Study 2 Data.xlsx', sheet_name='Property Data')

print("="*80)
print("USER PREFERENCES DATA")
print("="*80)
print(f"\nShape: {user_df.shape}")
print(f"\nColumns: {user_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(user_df.head())
print(f"\nData types:")
print(user_df.dtypes)
print(f"\nMissing values:")
print(user_df.isnull().sum())

print("\n" + "="*80)
print("PROPERTY CHARACTERISTICS DATA")
print("="*80)
print(f"\nShape: {prop_df.shape}")
print(f"\nColumns: {prop_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(prop_df.head())
print(f"\nData types:")
print(prop_df.dtypes)
print(f"\nMissing values:")
print(prop_df.isnull().sum())

# Save to file for easier viewing
with open('data_exploration.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("USER PREFERENCES DATA\n")
    f.write("="*80 + "\n\n")
    f.write(f"Shape: {user_df.shape}\n\n")
    f.write(f"Columns: {user_df.columns.tolist()}\n\n")
    f.write("First few rows:\n")
    f.write(user_df.head().to_string() + "\n\n")
    f.write("Data types:\n")
    f.write(user_df.dtypes.to_string() + "\n\n")
    f.write("Missing values:\n")
    f.write(user_df.isnull().sum().to_string() + "\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("PROPERTY CHARACTERISTICS DATA\n")
    f.write("="*80 + "\n\n")
    f.write(f"Shape: {prop_df.shape}\n\n")
    f.write(f"Columns: {prop_df.columns.tolist()}\n\n")
    f.write("First few rows:\n")
    f.write(prop_df.head().to_string() + "\n\n")
    f.write("Data types:\n")
    f.write(prop_df.dtypes.to_string() + "\n\n")
    f.write("Missing values:\n")
    f.write(prop_df.isnull().sum().to_string() + "\n")

print("\nData exploration saved to 'data_exploration.txt'")
