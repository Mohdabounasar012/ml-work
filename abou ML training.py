#!/usr/bin/env python
# coding: utf-8

# In[1]:


x=13
y=x*x
print(y)


# In[5]:


x=[2,3,4,5,6,7,8,9]
y=(len(x))
print(y)


# In[6]:


x=2
print(id(x))


# In[7]:


x=(1,2,3,4,5,6)
print(x[0])


# In[8]:


def main(x,y):
    y=x,y,x+y
    print(y)
main(0,1)


# In[18]:


x=-1 and 1
print(x)


# In[25]:


x=input('enter student name:')
y=int(input('enter student marks:'))
if y>200:
    print('pass')
elif y>=150:
    print('second')
else:
    print('fails')


# In[28]:


for i in range(6):
    print(i)


# In[34]:


x=[1,2,3,4,5]
for i in x:
    print(i)


# In[36]:


class Student:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def display(self):
        print('student name:',self.name)
        print('student age:',self.age)
s=Student('Noor',25)
s.display()


# In[41]:


import math
l=3
b=4
area=(l*b)
print(area)
    


# In[43]:


x=5
print(x)


# In[11]:


l1 = [1,2,3,]
l2 = [3,4,5]
l1+l2


# In[15]:


import pandas as pd
import numpy as np


# In[23]:


from numpy.random import randn
np.random.seed(301)


# In[24]:


df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())


# In[25]:


df


# In[26]:


df['W']


# In[27]:


df[['W','Z']]


# In[29]:


df[['W','X']]


# In[30]:


df.W


# In[31]:


type(df['W'])


# In[33]:


df['New']=df['W']+df['X']


# In[34]:


df


# In[36]:


df['New']=df['X']+df['Z']


# In[37]:


df


# In[38]:


df.drop('New',axis=1)


# In[39]:


df


# In[40]:


df.drop('New',axis=1,inplace=True)


# In[41]:


df


# In[42]:


df.drop('E',axis=0)


# In[43]:


df


# In[45]:


df.loc['A']


# In[46]:


df.iloc[2]


# In[47]:


df.loc['A','Y']


# In[49]:


df.loc[['A','B'],['W','X']]


# In[50]:


df


# In[51]:


#condition:
df>0


# In[52]:


df!=1


# In[53]:


df<0


# In[55]:


df<=1


# In[56]:


df>=1


# In[57]:


df>=0


# In[58]:


df<=0


# In[64]:


import numpy as np
import pandas as pd


# In[69]:


df = pd.DataFrame({'A':[934,56,5,634,np.nan],
                  'B':[5,13,14,np.nan,np.nan],
                  'C':[1,2,3,np.nan,np.nan],
                  'D':[np.nan,np.nan,np.nan,0,0],
                  'E':[np.nan,np.nan,2,435,465]})


# In[70]:


df


# In[71]:


df.head()


# In[74]:


df.tail(5)


# In[76]:


df.dropna()


# In[78]:


df.dropna(axis=1)


# In[79]:


df>0


# In[80]:


import matplotlib.pyplot as plt


# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


import numpy as np
x = np.linspace(0, 5, 11)
y = x ** 4


# In[87]:


x


# In[88]:


y


# In[89]:


plt.plot(x, y, 'r') # 'r' is the color red
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show()


# In[93]:


# plt.subplot(nrows, ncols, plot_number)
plt.subplot(1,2,1)
plt.plot(x, y, 'r--') # More on color options later
plt.subplot(1,2,2)
plt.plot(y, x, 'g*-')
plt.subplot(1,3,3)
plt.plot(x,y,'g*-');


# In[95]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(x, y, 'b')
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')


# In[96]:


# Creates blank canvas
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# Larger Figure Axes 1
axes1.plot(x, y, 'b')
axes1.set_xlabel('X_label_axes2')
axes1.set_ylabel('Y_label_axes2')
axes1.set_title('Axes 2 Title')

# Insert Figure Axes 2
axes2.plot(y, x, 'r')
axes2.set_xlabel('X_label_axes2')
axes2.set_ylabel('Y_label_axes2')
axes2.set_title('Axes 2 Title');


# In[97]:


# Use similar to plt.figure() except use tuple unpacking to grab fig and axes
fig, axes = plt.subplots()

# Now use the axes object to add stuff to plot
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# In[98]:


# Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2)


# In[100]:


for ax in axes:
    ax.plot(x, y, 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

# Display the figure object    
fig


# In[101]:


fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x, y, 'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

fig    
plt.tight_layout()


# In[102]:


fig, axes = plt.subplots(figsize=(12,3))

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# In[103]:


fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend()


# In[104]:


# Lots of options....

ax.legend(loc=1) # upper right corner
ax.legend(loc=2) # upper left corner
ax.legend(loc=3) # lower left corner
ax.legend(loc=4) # lower right corner

# .. many more options are available

# Most common to choose
ax.legend(loc=0) # let matplotlib decide the optimal location
fig


# In[105]:


# MATLAB style line color and style 
fig, ax = plt.subplots()
ax.plot(x, x**2, 'b.-') # blue line with dots
ax.plot(x, x**3, 'g--') # green dashed line


# In[106]:


fig, ax = plt.subplots()

ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant
ax.plot(x, x+2, color="#8B008B")        # RGB hex code
ax.plot(x, x+3, color="#FF8C00")        # RGB hex code 


# In[107]:


fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, x+1, color="red", linewidth=0.25)
ax.plot(x, x+2, color="red", linewidth=0.50)
ax.plot(x, x+3, color="red", linewidth=1.00)
ax.plot(x, x+4, color="red", linewidth=2.00)

# possible linestype options ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’
ax.plot(x, x+5, color="green", lw=3, linestyle='-')
ax.plot(x, x+6, color="green", lw=3, ls='-.')
ax.plot(x, x+7, color="green", lw=3, ls=':')

# custom dash
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...
ax.plot(x, x+ 9, color="blue", lw=3, ls='-', marker='+')
ax.plot(x, x+10, color="blue", lw=3, ls='--', marker='o')
ax.plot(x, x+11, color="blue", lw=3, ls='-', marker='s')
ax.plot(x, x+12, color="blue", lw=3, ls='--', marker='1')

# marker size and color
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 
        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="green");


# In[108]:


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight')
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60])
axes[2].set_xlim([2, 5])
axes[2].set_title("custom axes range");


# In[109]:


plt.scatter(x,y)


# In[110]:


from random import sample
data = sample(range(1, 1000), 100)
plt.hist(data)


# In[121]:


data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# rectangular box plot
plt.boxplot(data,vert=True,patch_artist=True);   


# In[122]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[123]:


tips=sns.load_dataset('tips')


# In[124]:


tips.head()


# In[125]:


sns.distplot(tips['total_bill'])


# In[129]:


sns.pairplot(tips)


# In[131]:


sns.pairplot(tips,hue='time',palette='coolwarm')


# In[133]:


sns.distplot(tips['total_bill'],kde=False,bins=40)


# In[134]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')


# In[135]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')


# In[136]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')


# In[141]:


# Don't worry about understanding this code!
# It's just for the diagram below
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Create dataset
dataset = np.random.randn(25)

# Create another rugplot
sns.rugplot(dataset);

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min,x_max,100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((5*dataset.std()**4)/(4*len(dataset)))**.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis,kernel,color = 'blue',alpha=0.5)

plt.ylim(0,1)


# In[ ]:




