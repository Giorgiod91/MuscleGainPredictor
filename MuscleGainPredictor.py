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

# function to get the weight(kg) cause that could also be a factor not sure yet
def get_weight(data):
    for column in data.columns:
        if '(kg)' in column.lower().strip():
            return column
    
    return None


# function to get energy intake kcal kg day that should be a major factor aswell 
def get_ernergy_intake(data):

    for column in data.columns:
        if "Energy" and "intake"in column.lower().strip():
            return column
        
    return None

# function to get the duration (weeks) thats also a major factor then later i will filter out to have the same weeks 
def get_weeks(data):

    for column in data.columns:
        if "Duration " and "(weeks)" in column.lower().strip():
            return column
        
    return None


# Get the relevant column names
protein = get_protein_intake(data)
LBM_change = get_LBM_change(data)
trainingTime = get_training_time(data)
weight= get_weight(data)

energy = get_ernergy_intake(data)
weeks = get_weeks(data)
# Print the column names for debugging
print(f"Protein intake column found: {protein}")
print(f"LBM change column found: {LBM_change}")
print(f"training time column found: {trainingTime}")
print(f"weight(kg) column found: {weight}")

print(f"energy column found: {energy}")
print(f"weeks column found: {weeks}")


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


if weight and LBM_change:
    x_data = data[weight]
    y_data = data[LBM_change]

    #plot data
    plt.scatter(x_data, y_data, c="green", label="data points")
    plt.xlabel(weight)
    plt.ylabel(LBM_change)
    plt.title(f"relationship between {weight} and {LBM_change}")
    plt.legend()
    plt.show()



if energy and LBM_change:

    x_data = data[energy]
    y_data = data[LBM_change]

    #plot data
    plt.scatter(x_data, y_data, c="yellow", label="data points")
    plt.xlabel(energy)
    plt.ylabel(LBM_change)
    plt.title(f"relationship between {energy} and {LBM_change}")
    plt.legend()
    plt.show()


if weeks and LBM_change:

    x_data = data[weeks]
    y_data = data[LBM_change]

    #plot data
    plt.scatter(x_data, y_data, c="red" , label="data poiint")
    plt.xlabel(weeks)
    plt.ylabel(LBM_change)
    plt.title(f"relationship between {weeks} and {LBM_change}")
    plt.legend()
    plt.show()

# picking good set of x feature to find a balance for the regression model

x_train = [protein, energy, trainingTime]
y_train = [LBM_change]



def compute_cost(X,y,w,b):
    # for a better understanding what happens 
    # f_wb = np.array([...])  # Predicted values VECTOR
    # y = np.array([...])  # True values VECTOR
    # errors = f_wb - y compute the vector of errors
    # cost = np.sum(errors ** 2) compute the squared errors and sum them


    m = X.shape[0]  # Number of training examples
    cost = 0.0 # initalize cost with 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) +b # predicted value of 1 example
        cost = cost + (f_wb_i - y[i]) **2
    cost = cost/ (2 * m) # Normalize errors and simplify gradients


    return cost
