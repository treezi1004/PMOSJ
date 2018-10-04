
# coding: utf-8

# # Part 2: Performance Metrics in Classifications
# 
# ### Chanil Park
# 
# ## Classification methods
# - kNN
# - Naive Bayes
# - SVM
# - Decision Tree
# - Random Forest
# - AdaBoost
# - Gradient Boosting
# - Linear Discriminant Analysis
# - Multi-layer Perceptron
# - Logistic Regression

# In[ ]:


# magic commands, sets the backend of matplotlib to the 'inline' backend
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#from utilities.losses import compute_loss
#from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Initialize seed
seed = 1000
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
test_size = 0.3

# Training settings
alpha = 0.9  # step size
max_iters = 100  # max iterations
tol = 0.1 #SGDRegressor, Ridge
verbose = 0


# In[ ]:


def load_data():
    # File Path is dependent on the starting directory path of Jupyter.
    # df = pd.read_csv("../data/adult.csv")
    train_data = pd.read_csv("../Part 2 Performance Metrics in Classification/data/adult.csv")
    test_data = pd.read_csv("../Part 2 Performance Metrics in Classification/data/adultTest.csv")
    return train_data, test_data


# In[ ]:


def createDummies(data):
    # Create dummy variables for categorical data
    cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data[var], prefix=var)
        datal=data.join(cat_list)
        data=datal
    # Determine the categorical columns
    data_vars = data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]
    # Removes categorical columns
    data_final = data[to_keep]
    data_final.columns.values
    return data_final


# In[ ]:


def cleanEducation(data):
    # simplify the categories
    data["education"]=np.where(data["education"] == '1st-4th', 'basic', data["education"])
    data["education"]=np.where(data["education"] == '5th-6th', 'basic', data["education"])
    data["education"]=np.where(data["education"] == '7th-8th', 'basic', data["education"])
    data["education"]=np.where(data["education"] == '9th', 'basic', data["education"])
    data["education"]=np.where(data["education"] == '10th', 'basic', data["education"])
    data["education"]=np.where(data["education"] == '11th', 'HighSchool', data["education"])
    data["education"]=np.where(data["education"] == '12th', 'HighSchool', data["education"])
    data["education"]=np.where(data["education"] == 'HS-grad', 'HighSchool', data["education"])
    data["education"].unique()
    return data


# In[ ]:


def standardize(train_data, test_data):
    """
    Standardize Data Set
    """
    #train_mean = train_data.mean()
    #train_std = train_data.std()
    #train_data = (train_data - train_mean) / train_std
    #test_data = (test_data - train_mean) / train_std
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


# In[593]:


def firstPreparation(data):
    """
    First preperation, drop the class and prepare data
    """
    data_full = data.copy()
    data = data.drop(["class"], axis = 1)
    labels = data_full["class"]
    labels = LabelEncoder().fit_transform(labels)
    return data_full, data, labels


# In[589]:


def data_preprocess(train_data, test_data):
    # drop class and prepare data(both train and test)
    train_data_full, train_data, train_labels = firstPreparation(train_data)
    test_data_full, test_data, test_labels = firstPreparation(test_data)
    # resize training data
    #train_data = pd.DataFrame(train_data.values.reshape(test_data.shape))
    train_data = cleanEducation(train_data)
    test_data = cleanEducation(test_data)
    # Handling categorized data
    train_data = createDummies(train_data)
    test_data = createDummies(test_data)
    #train_data = train_data.drop(["native-country_Holand-Netherlands"], axis = 1)
    # Standardize the inputs
    train_data, test_data = standardize(train_data, test_data)
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


# In[590]:


def predict(x, thetas):
    return x.dot(thetas)


# In[597]:


def applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels):
    start_time = datetime.datetime.now()  # Track learning starting time
    # prediction = cross_val_predict(classifier, train_data, test_data, cv=10)
    classifier.fit(train_data, train_labels)
    # print("Classifier fit is done")
    prediction = classifier.predict(test_data)
    # print("Prediction is done")
    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = round((end_time - start_time).total_seconds(), 2)  # Track execution time
    accuracy = round(accuracy_score(test_labels, prediction), 2)
    precision = round(precision_score(test_labels, prediction), 2)
    recall = round(recall_score(test_labels, prediction), 2)
    f1 = round(f1_score(test_labels, prediction), 2)
    print(classifier)
    print("Exection time: ", exection_time)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)  # R2 should be maximize
    print("F1: ", f1)
    print(classification_report(test_labels, prediction))
    print("\n")


# In[577]:


# check meaning of each columns
# data.columns


# In[578]:


# check whether there is missing values
# data.isnull().values.any()


# In[579]:


# have a look at data
# data.head()


# In[580]:


# have a look at kind of values
#for column in data.columns:
#    print(column, ":" , data[column].unique(), "\n")


# In[581]:


# check the categories of education
# data["education"].unique()


# In[582]:


# understand more about the data
# data.describe()


# In[583]:


# visualize the distributions of each columns (numerics)
# data.hist(bins=10,figsize=(14,10))
# plt.show()


# In[584]:


# look deeper into data by class
# data.groupby("class").mean()


# In[585]:


# have a look ata data by occupation
# data.groupby("occupation").mean()


# In[586]:


# have a look ata data by race
# data.groupby("race").mean()


# In[596]:


# load data
train_data, test_data = load_data();
# Preprocess the data
train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(train_data, test_data)
classifier = KNeighborsClassifier()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = GaussianNB()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = SVC()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = DecisionTreeClassifier()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = RandomForestClassifier()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = AdaBoostClassifier()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = GradientBoostingClassifier()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = LinearDiscriminantAnalysis()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = MLPClassifier()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)
classifier = LogisticRegression()
applyClassifierThenResult(classifier, train_data, train_labels, test_data_full, test_data, test_labels)

