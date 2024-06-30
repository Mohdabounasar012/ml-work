#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, svm
from sklearn.tree import DecisionTreeRegressor
import datetime as dt
from datetime import datetime


# In[2]:


df=pd.read_csv("C:\BostonHousing.csv")
df


# In[3]:


df.corr()


# In[4]:


df.isnull().sum()


# In[5]:


sns.heatmap(df.corr())


# In[6]:


sns.countplot(data=df,x=df.crim)


# In[7]:


sns.relplot(data=df,x='crim',y='medv',hue="zn",kind='line')


# In[8]:


sns.relplot(data=df,x='crim',y='medv',hue="zn")


# In[9]:


sns.relplot(data=df,x='crim',y='medv',hue="indus",kind='line')


# In[10]:


sns.relplot(data=df,x='crim',y='medv',hue="indus")


# In[11]:


sns.countplot(data=df,x=df.zn)


# In[12]:


sns.relplot(data=df,x='zn',y='medv',hue="indus")


# In[13]:


sns.countplot(data=df,x=df.indus)


# In[14]:


sns.relplot(data=df,x='chas',y='medv',hue="crim")


# In[15]:


sns.relplot(data=df,x='crim',y='medv',hue="nox")


# In[16]:


sns.relplot(data=df,x='zn',y='medv',hue="rm")


# In[17]:


sns.relplot(data=df,x='zn',y='medv',hue="rm",kind='line')


# In[18]:


sns.relplot(data=df,x='crim',y='medv',hue="age")


# In[19]:


sns.relplot(data=df,x='zn',y='medv',hue="dis") 


# In[20]:


sns.relplot(data=df,x='crim',y='medv',hue="dis")


# In[21]:


sns.relplot(data=df,x='indus',y='medv',hue="dis")


# In[22]:


sns.relplot(data=df,x='zn',y='medv',hue="rad")


# In[23]:


sns.relplot(data=df,x='indus',y='medv',hue="rad")


# In[24]:


sns.relplot(data=df,x='rad',y='medv',hue="crim")


# In[25]:


sns.relplot(data=df,x='crim',y='medv',hue="rad")


# In[26]:


sns.relplot(data=df,x='tax',y='medv',hue="zn")


# In[27]:


sns.relplot(data=df,x='zn',y='medv',hue="tax")


# In[28]:


sns.relplot(data=df,x='tax',y='medv',hue="indus")


# In[29]:


sns.relplot(data=df,x='indus',y='medv',hue="tax")


# In[30]:


sns.relplot(data=df,x='zn',y='medv',hue="ptratio")


# In[31]:


sns.relplot(data=df,x='indus',y='medv',hue="ptratio")


# In[32]:


sns.relplot(data=df,x='crim',y='medv',hue="ptratio")


# In[33]:


sns.relplot(data=df,x='zn',y='medv',hue="b")


# In[34]:


sns.relplot(data=df,x='indus',y='medv',hue="b")


# In[35]:


sns.relplot(data=df,x='crim',y='medv',hue="b")


# In[36]:


sns.relplot(data=df,x='ptratio',y='medv',hue="b")


# In[37]:


sns.relplot(data=df,x='age',y='medv',hue="b")


# In[38]:


sns.relplot(data=df,x='zn',y='medv',hue="lstat")


# In[39]:


sns.relplot(data=df,x='indus',y='medv',hue="lstat")


# In[40]:


sns.relplot(data=df,x='age',y='medv',hue="lstat")


# In[41]:


sns.relplot(data=df,x='lstat',y='medv',hue="crim")


# In[42]:


sns.relplot(data=df,x='crim',y='medv',hue="lstat")


# In[ ]:




