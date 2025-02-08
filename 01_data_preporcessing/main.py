import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Data.csv")
x_features = dataset.iloc[:, :-1].values  # get all columns except last
y_dependent_var = dataset.iloc[:, -1].values # get last column

print(x_features)
print(y_dependent_var)

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x_features[:, 1:3])
x_features[:, 1:3] = imputer.transform(x_features[:, 1:3])
print("x_features: \n", x_features)

#encoding countries with one hot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

col_transformer_obj = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x_features = np.array(col_transformer_obj.fit_transform(x_features))
print(x_features)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_dependent_var = le.fit_transform(y_dependent_var)
print(y_dependent_var)

# Spliting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_dependent_var, test_size=0.2, random_state=1)
print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)