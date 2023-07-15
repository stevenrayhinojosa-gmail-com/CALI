#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Read the Excel file
df = pd.read_excel(r'C:\Users\steve\Documents\JohnDoe.xlsx', usecols="C:Z", skiprows=range(1, 4), nrows=4)

df = df.drop('Red/Yellow/Green Data', axis=1)
df.columns


# In[2]:


X = df.drop('Unnamed: 25', axis=1)  # Drop the 'Sheet35' column from features
y = df['Unnamed: 25'] 


# In[3]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X)
print(y)
# Create and train a machine learning model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




