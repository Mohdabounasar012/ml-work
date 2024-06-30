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


df=pd.read_csv(r"C:\nasar dataset\loan_borowwer_data.csv")
df


# In[3]:


df.corr()


# In[4]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# In[5]:


df.describe()
df


# In[6]:


sns.relplot(data=df,x='int.rate',y='not.fully.paid',hue="installment",kind='line')


# In[7]:


sns.relplot(data=df,x='purpose',y='not.fully.paid',hue="installment",kind='line')


# In[8]:


df.isnull().sum()


# In[9]:


df1=df.drop("purpose", axis='columns')
df1


# In[10]:


x=df1.iloc[:,1:12]
y=df1[['not.fully.paid']]


# In[11]:


x.head(5)


# In[12]:


y.head(5)


# In[13]:


y.tail(5)


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[15]:


x_train


# In[16]:


x_test


# In[17]:


y_train


# In[18]:


y_test


# In[19]:


from sklearn import tree


# In[20]:


tree_model=tree.DecisionTreeClassifier()


# In[21]:


tree_model.fit(x_train,y_train)


# In[22]:


y_pred=tree_model.predict(x_test)


# In[23]:


y_pred


# In[24]:


accuracy=tree_model.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[25]:


from sklearn.metrics import classification_report


# In[26]:


print('classification_score')
print(classification_report(y_test,y_pred))


# In[27]:


y_test


# In[28]:


print(len(y_pred))


# In[29]:


plt.scatter(x=y_test,y=y_pred)


# In[30]:


cm=confusion_matrix(y_test,y_pred)


# In[31]:


cm


# In[32]:


sns.heatmap(cm,
           annot=True,
           fmt='g')
plt.ylabel('y_pred',fontsize=13)
plt.xlabel('y_test',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[33]:


pred=tree_model.predict(x)
pred


# In[34]:


print(classification_report(y,pred))


# In[35]:


cf=confusion_matrix(y,pred)
cf


# In[36]:


sns.heatmap(cf,
           annot=True,
           fmt='g',
           xticklabels=[7725,1241],
           yticklabels=[320,292])
plt.ylabel('pred',fontsize=13)
plt.xlabel('y',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


Random_model=RandomForestClassifier()


# In[39]:


Random_model.fit(x_train,y_train)


# In[40]:


y_pred1=Random_model.predict(x_test)


# In[41]:


y_pred1


# In[42]:


accuracy=Random_model.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[43]:


x=y_pred
y=y_pred1
plt.plot(x,y)


# In[44]:


from sklearn.naive_bayes import GaussianNB


# In[45]:


Gaussian_model=GaussianNB()


# In[46]:


Gaussian_model.fit(x_train,y_train)


# In[47]:


accuracy=Gaussian_model.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[48]:


pred2=Gaussian_model.predict(x_test)
pred2


# In[49]:


from sklearn.naive_bayes import CategoricalNB


# In[50]:


cnb_model=CategoricalNB()


# In[51]:


cnb_model.fit(x_train,y_train)


# In[ ]:




