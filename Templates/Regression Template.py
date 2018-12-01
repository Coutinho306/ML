
# coding: utf-8

# # Regression Templates

# ### Importing Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# ### Setting Datasets directory and Importing dataset 

# In[ ]:


os.chdir("C:\\Users\\Thiago\\Desktop\\Python-ML\\Datasets")

dataset = pd.read_csv("Position_Salaries.csv")


# ### Creating features and target variables

# In[ ]:


X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# ### Splitting in Training and Test set 

# In[ ]:



from sklearn.model_selection import train_test_split
X_treino,X_teste,y_treino,y_teste = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Fitting the Regression model

# ### Predicting Results

# In[ ]:


y_pred = regressor.predict()


# ### Visualiazing Regression Results

# In[ ]:


plt.figure(figsize = (10,7))

plt.scatter(X, y , color ="red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# ### Visualiazing Regression Results in Higher Resolution

# In[ ]:


X_grid = np.arrange(range(min(X), max(X), 0.1))
X_grid = np.reshape(len(X_grid), 1)

plt.figure(figsize = (10,7))

plt.scatter(X, y , color ="red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

