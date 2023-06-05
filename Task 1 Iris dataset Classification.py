#!/usr/bin/env python
# coding: utf-8

# In[13]:


print("""The aim is to classify Iris flower into its three species setosa, virginica and versicolor based on their sepal and petal
measurements.\n
Input characteristics are:
-Sepal length
-Sepal width
-Petal length
-Petal width\n\n
Importing all essential libraries.""")


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


columns = ['Sepal length' , 'Sepal width' , 'Petal length' , 'Petal width' , 'Species']
#Loading iris flower dataset
print("Iris dataset")
df= pd.read_csv('iris.data', names=columns)
df.head()


# In[26]:


#Display
print("Display of statistics of dataset")
df.describe()


# In[27]:


# Information of datatype
df.info()


# In[28]:


#Data Preprocessing
df.isnull().sum()


# In[38]:


print("Display using Histograms")


# In[32]:


#display of column Sepal length
df['Sepal length'].hist()


# In[37]:


#display of column Sepal width
df['Sepal width'].hist()


# In[36]:


#display of column Petal length
df['Petal length'].hist()


# In[35]:


#display of column Petal width
df['Petal width'].hist()


# In[50]:


#display using scatterplot
colors = ['green' , 'orange', 'blue']
species = ['Iris-setosa' , 'Iris-versicolor' , 'Iris-virginica']


# In[51]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['Sepal length'] , x['Sepal width'] , c = colors[i], label= species[i])
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.legend()


# In[52]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['Petal length'] , x['Petal width'] , c = colors[i], label= species[i])
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend()


# In[53]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['Sepal length'] , x['Petal length'] , c = colors[i], label= species[i])
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")
    plt.legend()


# In[54]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['Sepal width'] , x['Petal width'] , c = colors[i], label= species[i])
    plt.xlabel("Sepal width")
    plt.ylabel("Petal width")
    plt.legend()


# In[55]:


#Correlation matrix
df.corr()


# A correlation matrix is a statistical technique used to evaluate the relationship between two variables in a data set. The matrix is a table in which every cell contains a correlation coefficient, where 1 is considered a strong relationship between variables, 0 a neutral relationship and -1 a not strong relationship.

# In[71]:


corr = df.corr()
fig , ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot = True, ax = ax , cmap = 'hot')


# In[74]:


#label encoder assigns numeric value to output from 0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[76]:


df['Species'] = le.fit_transform(df['Species'])
df.head(150)


# In[85]:


#Training and Testing data
#Train = 75
#Test = 25
from sklearn.model_selection import train_test_split
X = df.drop(columns= ['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X ,Y , test_size= 0.25)


# In[86]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[87]:


model.fit(x_train , y_train)


# In[89]:


#print metric to gain performance of Logistic Regression
print("Accuracy:" ,model.score(x_test , y_test)*100)


# In[90]:


#KNN 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[91]:


model.fit(x_train , y_train)


# In[92]:


#print metric to gain performance of KNN
print("Accuracy:" ,model.score(x_test , y_test)*100)


# In[93]:


#Decision tree 
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[94]:


model.fit(x_train , y_train)


# In[95]:


#print metric to gain performance of Decision tree
print("Accuracy:" ,model.score(x_test , y_test)*100)


# In[96]:


#SVM
from sklearn.svm import SVC
model = SVC()


# In[97]:


model.fit(x_train , y_train)


# In[98]:


#print metric to gain performance of SVM
print("Accuracy:" ,model.score(x_test , y_test)*100)


# In[100]:


print("""Hence we successfully learnt to train a machine learning model and 
perform classification of species based on input measurement characteristics 
using the Iris dataset.""")

