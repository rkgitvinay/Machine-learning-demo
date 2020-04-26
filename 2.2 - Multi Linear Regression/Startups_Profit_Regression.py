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

