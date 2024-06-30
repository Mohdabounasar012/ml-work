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
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix


# In[2]:


df=pd.read_csv("C:\data.csv")
df


# In[3]:


conditions = [
    (df['diagnosis'] == 'M'),
    (df['diagnosis'] == 'B'), 
]
choices = [1,0]
df["diagnosis_values"] = np.select(conditions, choices)
df


# In[4]:


df.corr()


# In[5]:


sns.heatmap(df.corr())


# In[6]:


df.isnull().sum()


# In[7]:


x=df.iloc[:,2:32]
y=df[['diagnosis_values']]


# In[8]:


x.head(5)


# In[9]:


y.head(5)


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=5)


# In[24]:


x_train


# In[25]:


x_test


# In[26]:


y_train


# In[27]:


y_test


# In[28]:


from sklearn import tree


# In[29]:


tree_model=tree.DecisionTreeClassifier()


# In[30]:


tree_model.fit(x_train,y_train)


# In[31]:


y_pred=tree_model.predict(x_test)


# In[32]:


accuracy=tree_model.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[33]:


print('classification_score')
print(classification_report(y_test,y_pred))


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


y_test


# In[36]:


print(len(y_pred))


# In[37]:


plt.scatter(x=y_test, y=y_pred)


# In[42]:


cf=confusion_matrix(y_test ,y_pred)
cf


# In[43]:


pred=tree_model.predict(x)


# In[44]:


print(classification_report(y,pred))


# In[45]:


cm=confusion_matrix(y,pred)
cm


# In[ ]:




