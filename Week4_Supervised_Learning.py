#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# 
# ## Churn Prediction Analysis

# In this assignment, we are going to predict customer churn for a telecom company using the K-Nearest Neighbor algorithm.

# # Setup
# 
# Import python libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

get_ipython().magic('matplotlib inline')


# # Exercise 1
# 
# Load the csv file called 'Churn.csv' using pandas and assign it to a variable called df. (hint: pd.read_csv?) Look at the first 10 rows of the data set. (hint: df.head?)

# In[2]:


data = pd.read_csv('Churn.csv')
data.head(10)


# # Exercise 2: 
# 
# Which column indicates whether the user has churned? How many users are there in your dataset? (hint: len?)

# In[ ]:


churned = data.loc[:,data.columns == 'churned']
churned


# # Exercise 3:
# 
# Use df.describe() to explore each column. Why is the count different for each column and not equal to 5000?

# In[ ]:


data.describe()


# # Exercise 4: 
# 
# Fill the missing numbers with the median of the whole dataset! (hint: df.fillna?) Check to see if the counts for all columns is now 5000.

# In[10]:


data.fillna(data.mean(), inplace=True)
# data.head(10)

dataCopy = data
dataCopy


# # Exercise 5:
# 
# Separate the data into the features and labels. Assign the features to variable X and labels to variable y.

# In[23]:


x = dataCopy.loc[:, data.columns != 'churned']
y = dataCopy.loc[:, data.columns == 'churned']


# # Exercise 6: 
# 
# Split the data into 70% training set and 30% test set and assign them to variables named X_train, X_test, y_train, y_test (hint: train_test_split?)

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=45)
X_train


# # Exercise 7: 
# 
# Create a N-Nearest Neighbors classifier of 5 neighbors. (hint: KNeighborsClassifier?)

# In[42]:


KNN = KNeighborsClassifier(n_neighbors=6)


# # Exercise 8:
# 
# Fit the model to the training set. (hint: knn.fit?)

# In[43]:


model = KNN.fit(X_train, y_train.values.ravel())


# # Exercise 9:
# 
# Use the model to make a prediction on the test set. Assign the prediction to a variable named y_pred

# In[44]:


y_pred = model.predict(X_test)


# # Exercise 10:
# 
# Determine how accurate your model is at making predictions. (hint: accuracy_score?)

# In[45]:


accuracy_score(y_test,y_pred,normalize=True, sample_weight=None)


# # Exercise 11 (Optional)
# 
# Try different number of k neighbors between 1 and 10 and see if you find a better result

# In[ ]:




