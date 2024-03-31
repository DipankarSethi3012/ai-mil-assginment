#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

titanic_df = pd.read_csv('Titanic-Dataset.csv')


# In[2]:


titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)


# In[3]:


titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)


# In[4]:


X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[7]:


def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))


# In[8]:


def cost(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    return -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


# In[9]:


def gradient(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    return 1 / m * np.dot(X.T, (h - y))


# In[10]:


def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    costs = []
    for _ in range(epochs):
        theta -= learning_rate * gradient(X, y, theta)
        costs.append(cost(X, y, theta))
    return theta, costs


# In[11]:


X_train_bias = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_bias = np.c_[np.ones((len(X_test), 1)), X_test]


# In[12]:


theta, costs = gradient_descent(X_train_bias, y_train)


# In[13]:


def predict(X, theta):
    return np.round(hypothesis(X, theta)).astype(int)


# In[14]:


y_pred = predict(X_test_bias, theta)


# In[15]:


accuracy = np.mean(y_pred == y_test) * 100
print("Accuracy:", accuracy, "%")

