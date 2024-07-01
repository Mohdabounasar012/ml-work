#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df=pd.read_csv("D:\insurance.csv")
df


# In[3]:


df.corr()


# In[4]:


sns.heatmap(df.corr())


# In[5]:


sns.relplot(data = df, x = "age", y = "charges", hue = 'smoker', kind = "line" )


# In[44]:


sns.relplot(data = df, x = "region", y = "charges",hue="smoker")


# In[7]:


sns.relplot(data = df, x = 'children', y = 'charges')


# In[8]:


sns.relplot(data = df, x = 'children', y = 'charges', hue = 'age', style = 'smoker')


# In[9]:


sns.pairplot(df)


# In[10]:


conditions = [
    (df['smoker'] == 'yes'),
    (df['smoker'] == 'no'), 
]
choices = [1,0]
df["smoker_values"] = np.select(conditions, choices)
df


# In[11]:


x = df[['age', 'smoker_values']]
y = df[['charges']]


# In[12]:


x_train,x_test, y_train,y_test = train_test_split(x,y, train_size = 0.7, random_state = 5)


# In[13]:


x_train


# In[14]:


x_test


# In[15]:


y_test


# In[16]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[17]:


predicted=reg.predict(x_test)
expected=y_test
print(predicted)


# In[20]:


plt.Line2D(x_train,x_test, )
 


# In[27]:


sns.relplot(data=df,x="region",y="sex",hue="smoker",kind="line")


# In[46]:


plt.scatter(data=df,x="charges",y="age")
plt.show()


# In[47]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[48]:


print('Intercept:\n',reg.intercept_)
print('Coefficients:\n',reg.coef_)


# In[49]:


accuracy=reg.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[50]:


from sklearn import tree


# In[52]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model


# In[53]:


tree_model= DecisionTreeRegressor()
tree_model.fit(x_train,y_train)


# In[54]:


y_pred=tree_model.predict(x_test)


# In[55]:


accuracy=tree_model.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[56]:


y_pred=reg.predict(x_test)


# In[58]:


predicted1=reg.predict(x_train)
expected=y_train
print(predicted1)


# In[ ]:




