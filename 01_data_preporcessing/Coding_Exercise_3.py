import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("titanic.csv")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
categorical_features = ['Sex', 'Embarked', 'Pclass']
col_transformer_obj = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X = col_transformer_obj.fit_transform(dataset)
X= np.array(X)

print(X)

le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])

print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)

