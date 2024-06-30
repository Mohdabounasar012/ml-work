#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score


# In[3]:


df=pd.read_csv("D:\cereal.csv")
df


# In[4]:


x=df.iloc[:,3:15]

x


# In[5]:


df.corr()


# In[6]:


sns.pairplot(df)


# In[7]:


x = df[['sugars' and 'vitamins']]
y = df['type']
plt.title('Sugars & Vitamins')
plt.xlabel('Sugars & Vitamins')
plt.ylabel('Types')
plt.show()


# In[10]:


df.plot(x="type", y=["sugars", "vitamins"], kind="bar", figsize=(9, 6))


# In[47]:


df


# In[49]:


conditions=[
    (df['mfr']=='N'),
    (df['mfr']=='Q'),
    (df['mfr']=='K'),
    (df['mfr']=='R'),
    (df['mfr']=='G'),
    (df['mfr']=='P'),
    (df['mfr']=='A')
]
choices=['Nabisco','Quaker Oats','Kelloggs','Raslston Purina', 'General Mills' , 'Post' , 'American Home Foods Products']
df['MFR_FUL_NAME']=np.select(conditions,choices)


# In[50]:


df


# In[51]:


df.groupby(['mfr'])['type'].count()


# In[52]:


sns.countplot(data=df,x=df.mfr)


# In[57]:


sns.heatmap(df.corr())


# In[ ]:




