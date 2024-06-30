#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB


# 

# In[4]:


df = pd.read_csv(r"D:\voice-classification.csv")
df


# In[5]:


df.corr()


# In[6]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')


# In[7]:


df.isnull().sum()


# In[8]:


sns.relplot(data=df,x='sfm',y='centroid',col='label',hue='maxdom')


# In[9]:


df = df.drop(['centroid', 'maxdom','dfrange', 'sd', 'IQR','Q25'], axis = True)
df


# In[10]:


df = df.drop(['meanfreq', 'median'], axis = 1)
df


# In[11]:


x = df.iloc[: , 0:12]
y = df.iloc[:,-1:]


# In[12]:


x


# In[13]:


y = df.iloc[:,-1:]
y


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.75, random_state= 42)


# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


rc = RandomForestClassifier()


# In[17]:


y_train


# In[18]:


rc.fit(x_train, y_train)


# In[19]:


y_pred = rc.predict(x_test)


# In[20]:


accuracy1 = rc.score(x_test, y_test)
print("Accuracy:", accuracy1)


# In[21]:


print('classification_score')
print(classification_report(y_test, y_pred))


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


LC = LogisticRegression()


# In[24]:


LC.fit(x_train,y_train)


# In[25]:


y_pred1 = LC.predict(x_test)


# In[26]:


accuracy2 = rc.score(x_test, y_test)
print("Accuracy:", accuracy2)


# In[27]:


print('classification_score')
print(classification_report(y_test, y_pred))


# In[ ]:




