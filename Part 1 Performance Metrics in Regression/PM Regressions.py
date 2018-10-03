
# coding: utf-8

# In[ ]:


# Part 1: Performance Metrics in Regression

### Chanil Park

## regression methods
- linear regression
- k-neighbors regression
- Ridge regression
- decision tree regression
- random forest regression
- gradient Boosting regression
- SGD regression
- support vector regression (SVR)
- linear SVR
- multi-layer perceptron regression.


# In[178]:


# magic commands, sets the backend of matplotlib to the 'inline' backend
get_ipython().run_line_magic('matplotlib', 'inline')


# In[179]:


# -*- coding: utf-8 -*-

"""

Performance Metrics in Regression
x = weight and y = height.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#from utilities.losses import compute_loss
#from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# rms = sqrt(mean_squared_error(y_actual, y_predicted))

# General settings
#from utilities.visualization import visualize_train, visualize_test


# In[244]:


# Initialize seed
seed = 1000
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.9  # step size
max_iters = 100  # max iterations
tol = 0.1 #SGDRegressor, Ridge
verbose = 0


# In[221]:


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
    # File Path is dependent on the starting directory path of Jupyter.
    # df = pd.read_csv("../data/diamonds.csv")
    df = pd.read_csv("../Part 1 Performance Metrics in Regression/data/diamonds.csv")
    return df


# In[222]:


def standardize(train_data, test_data):
    """
    Standardize Data Set
    """
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std
    return train_data, test_data
    


# In[223]:


def preprocessing(data):
    """
    Pre-processing the data
    """
    data_full = data.copy()
    data = data.drop(["price"], axis = 1)
    labels = data_full["price"]
    return data_full, data, labels


# In[224]:


def data_preprocess(data):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param data: the given dataset (format: panda DataFrame)
    :return: train_data       train data contains only inputs
             train_labels     train data contains only labels
             test_data        test data contains only inputs
             test_labels      test data contains only labels
             train_data_full       train data (full) contains both inputs and labels
             test_data_full       test data (full) contains both inputs and labels
    """
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size = train_test_split_test_size, random_state=seed)

    # Pre-process data (both train and test)
    train_data_full, train_data, train_labels = preprocessing(train_data)
    test_data_full, test_data, test_labels = preprocessing(test_data)
    
    # Handling categorized data
    train_data = pd.get_dummies(train_data, columns=['cut', 'color', 'clarity'])
    # print(train_data.head())
    test_data = pd.get_dummies(test_data, columns=['cut', 'color', 'clarity'])    
    # print(test_data.head())
    
    # Standardize the inputs
    train_data, test_data = standardize(train_data, test_data)

    # Tricks: add dummy intercept to both train and test
    train_data['intercept_dummy'] = pd.Series(1.0, index = train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index = test_data.index)
    # print(train_data.head())
    # print(test_data.head())    
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


# In[225]:


def predict(x, thetas):
    return x.dot(thetas)


# In[234]:


def applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels):
    start_time = datetime.datetime.now()  # Track learning starting time
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    #pred_label = predict(test_data, gradient_ws[-1])
    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = round((end_time - start_time).total_seconds(), 2)  # Track execution time
    mse = round(mean_squared_error(test_labels, prediction), 2)
    rmse = round(np.sqrt(mse), 2)
    r2_error = round(r2_score(test_labels, prediction), 2)
    mae = round(mean_absolute_error(test_labels, prediction), 2)
    print(model)
    print("Exection time: ", exection_time)
    #print("Coefficients: ", model.coef_)
    #print("Intercept: ", model.intercept_)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2: ", r2_error)  # R2 should be maximize
    print("MAE: ", mae)
    print("\n")
    #plt.scatter(x=test_data_full["carat"], y=test_data_full["price"], color='blue')
    #plt.plot(test_data_full["carat"], pred_label, color='red', linewidth=2)


# In[245]:


if __name__ == '__main__':
    # load data 
    data = load_data()

    # Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)

    # Build baseline model
    model = LinearRegression()
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = KNeighborsRegressor()
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = Ridge(max_iter=max_iters, alpha=alpha, tol=tol)
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = DecisionTreeRegressor()
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = RandomForestRegressor()
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = GradientBoostingRegressor(alpha=alpha)
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = SGDRegressor(max_iter=100, alpha=alpha, tol=0.0001)
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = SVR(max_iter=max_iters, tol=tol)
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = LinearSVR(max_iter=max_iters, tol=tol)
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    model = MLPRegressor(max_iter=max_iters, alpha=alpha, tol=tol)
    applyModelThenResult(model, train_data, train_labels, test_data_full, test_data, test_labels)
    

