## Data Preprocessings

# STEP 1 - Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# STEP 2 -  Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1] # take all cols except last one
y = dataset.iloc[:, 3] # take last col

# STEP 5 - Splittig dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# STEP 6 - Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""