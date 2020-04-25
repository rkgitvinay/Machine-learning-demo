# Simple Linear Regression Model 
# predict Salary Based on Years of Experience 

# STEP 1 - Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# STEP 2 -  Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # take all cols except last one (Independent Cols)
y = dataset.iloc[:, 1].values # take Dependent Col

# STEP 3 - Splittig dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# STEP 4 - fitting simple linear regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # train regressor variable with training data

# STEP 5 - Predicting the Test set result
y_predicted = regressor.predict(X_test)

# STEP 6 - Visualising training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()

# STEP 6 - Visualising Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()
