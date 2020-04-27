# Multiple Linear Regression
# Predict which startp will produce higher profit

## Import Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



## Import Datasets
dataset = pd.read_csv('Startups_Profit.csv')
#Select Independent Variables (R&D spend, Admin, marketing spend and state)
X = dataset.iloc[:, :-1]
# Select Dependent Variable (Profit)
Y = dataset.iloc[:, 4]



## Encodeing Categorical data
## Encode Independent Variable (State)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
# LabelEncoder is used to chagne the text column (State) into numbers
X.iloc[:, 3] = labelEncoder_X.fit_transform(X.iloc[:, 3])
# OneHotEncoder is used to create different columns for differerent categories (New York, Florida, California)
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()


## Avoiding the Dummy Variable Trap
# Remove the first dummy variable (State)
X = X[:, 1:]


## Spliting the dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


## Fitting multilinear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predicting the Test set Results
y_predicted = regressor.predict(X_test)

# Optimize the model Using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis=1)

X_optimal = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()# OLS = Ordinary Least Squares
regressor_OLS.summary()
#exclude the variable which has highest P value Or P < LS (0.05)
X_optimal = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_OLS.summary()

X_optimal = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_OLS.summary()

X_optimal = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_OLS.summary()

X_optimal = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_OLS.summary()

