import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_set = pd.read_csv("Salary_Data.csv")
# X = data_set["YearsExperience"]
# y = data_set["Salary"]
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values
# Split training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_test)

# Train simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the test set results
y_predict = regressor.predict(X_test)

# Visualising training set results
plt.scatter(X_train, y_train, color= 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('salary vs experience (training set)')

plt.xlabel('years of exp')
plt.ylabel('salary')

# plt.show()
plt.savefig("output_plotTraining.png")
plt.clf()

# Visualising test results 
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience (test set)')

plt.xlabel('years of exp')
plt.ylabel('salary')

# plt.show()
plt.savefig("output_plotTest.png")
plt.clf()

## Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
