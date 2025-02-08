import pandas as pd

data_set = pd.read_csv('housing.csv')
print(data_set.head())
print(data_set.info())

data_set.hist(bins=50, figsize=(20,15))
