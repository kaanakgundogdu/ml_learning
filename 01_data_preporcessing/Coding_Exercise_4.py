# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

# Load the Iris dataset

dataset = pd.read_csv('iris.csv')

# Separate features and target

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



# Split the dataset into an 80-20 training-test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

sc = StandardScaler()

# Apply feature scaling on the training and test sets
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Print the scaled training and test sets

print(X_train)
print(X_test)


