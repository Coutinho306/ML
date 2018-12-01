
# coding: utf-8

# # Importing Packages

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Exploring Data

# In[2]:


# Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[4]:


cancer.keys()


# In[7]:


print(cancer['DESCR'])


# In[20]:


print(cancer['data'])


# In[11]:


print(cancer['feature_names'])


# In[10]:


print(cancer['target'])


# In[13]:


print(cancer['target_names'])


# In[14]:


cancer['data'].shape


# # Creating a dataframe with the entire data + the target in the same dataset, and renaming the columns

# In[17]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[19]:


df_cancer.head()


# In[21]:


df_cancer.tail()


# In[22]:


df_cancer.describe()


# # Visualizing Data
# 

# In[24]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius','mean texture','mean perimeter','mean area','mean smoothness'])


# In[26]:


sns.countplot(df_cancer['target'])


# In[27]:


sns.scatterplot(data= df_cancer, x = 'mean area', y= 'mean smoothness', hue = 'target')


# In[32]:


plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)


# # Training Model

# In[40]:


#Dividing in Features and Target Variables
X = df_cancer.drop(['target'], axis = 1)


# In[44]:


y = df_cancer['target']


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[49]:


X_train


# In[50]:


y_train


# In[51]:


from sklearn.svm import SVC


# In[52]:


from sklearn.metrics import classification_report, confusion_matrix


# In[53]:


svc_model = SVC()


# In[55]:


# Trainning the model
classifier = svc_model.fit(X_train,y_train)


# # EVALUATING THE MODEL

# In[57]:


#Predicting the Results
y_pred = classifier.predict(X_test)


# In[58]:


y_pred


# In[60]:


cm = confusion_matrix(y_test, y_pred)


# In[65]:


sns.heatmap(cm, annot = True )


# # IMPROVING THE MODEL - NORMALIZATION

# In[67]:


# Applying Feature Scaling - Trainning Set


# In[69]:


min_train = X_train.min()


# In[70]:


range_train = (X_train-min_train).max()


# In[71]:


X_train_scaled = (X_train-min_train)/range_train


# In[74]:


sns.scatterplot(x= X_train['mean area'],y = X_train['mean smoothness'], hue= y_train)


# In[75]:


sns.scatterplot(x= X_train_scaled['mean area'],y = X_train_scaled['mean smoothness'], hue= y_train)


# In[ ]:


# Applying Feature Scaling - Test Set


# In[76]:


min_test = X_test.min()


# In[77]:


range_test = (X_test-min_test).max()


# In[78]:


X_test_scaled = (X_test-min_test)/range_test


# In[80]:


# Trainning the model with Normalization


# In[82]:


classifier = svc_model.fit(X_train_scaled,y_train)


# In[83]:


#Predicting the Results 
y_pred = classifier.predict(X_test_scaled)


# In[85]:


cm = confusion_matrix(y_test, y_pred)


# In[86]:


sns.heatmap(cm, annot = True )


# In[88]:


# Print Classificatio Reports


# In[89]:


print(classification_report(y_test,y_pred))


# # IMPROVING THE MODEL - GRID SEARCH

# In[92]:


param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001], 'kernel': ['rbf']}


# In[93]:


from sklearn.model_selection import GridSearchCV


# In[95]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)


# In[96]:


grid.fit(X_train_scaled,y_train)


# In[97]:


grid.best_params_


# In[99]:


grid_pred = grid.predict(X_test_scaled)


# In[102]:


cm = confusion_matrix(y_test, grid_pred)


# In[103]:


sns.heatmap(cm, annot=True)


# In[104]:


print(classification_report(y_test,grid_pred))

