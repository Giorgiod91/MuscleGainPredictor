import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("dataOne.csv")

# Print the first few rows and the columns to check the data
print("First few rows of the dataset:")
print(data.head())
print("Columns in the dataset:")
print(data.columns)

# Function to get the column name for protein intake
def get_protein_intake(data):
    for column in data.columns:
        # Perform a case-insensitive search and strip extra spaces
        if "protein intake (g/kg/day)" in column.lower().strip():
            return column
    return None  

# Function to get the column name for LBM change
def get_LBM_change(data):
    for column in data.columns:
        if "lbm change per week (kg/wk)" in column.lower().strip(): 
            return column
    return None 

# function to get the training time times per week cause thats a big factor in my opinion
def get_training_time(data):
    for column in data.columns:
        if "(times/week)" in column.lower().strip():
            return column
        
    return None





# Get the relevant column names
protein = get_protein_intake(data)
LBM_change = get_LBM_change(data)
trainingTime = get_training_time(data)

# Print the column names for debugging
print(f"Protein intake column found: {protein}")
print(f"LBM change column found: {LBM_change}")
print(f"training time column found: {trainingTime}")

# Check if the columns were found
if protein and LBM_change:
    # Extract the data for plotting
    x_data = data[protein]
    y_data = data[LBM_change]

    # Plotting the data
    plt.scatter(x_data, y_data, c="blue", label="data points")
    plt.xlabel(protein)
    plt.ylabel(LBM_change)
    plt.title(f'{protein} and the resulting {LBM_change}')
    plt.legend()
    plt.show()


if trainingTime and LBM_change:

    x_data = data[trainingTime]
    y_data = data[LBM_change]

    #plotting the data
    plt.scatter(x_data, y_data, c="red", label="data points")
    plt.xlabel(trainingTime)
    plt.ylabel(LBM_change)
    plt.title(f"{trainingTime} and the resulting {LBM_change}")
    plt.legend()
    plt.show()
