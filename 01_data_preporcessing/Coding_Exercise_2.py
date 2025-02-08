# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 

# Load the dataset

dataset = pd.read_csv('pima-indians-diabetes.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("features", x)
# print(y)
# Identify missing data (assumes that missing data is represented as NaN)

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, :])
x_fit = imputer.transform(x[:,:])

print("fit", x)