# --------------------------REGRESSION TEMPLATE---------------------

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting Datasets directory
os.chdir("C:\\Users\\Thiago\\Desktop\\Python-ML\\Datasets")

# Importing dataset 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Splitting in Training and Test set 
from sklearn.model_selection import train_test_split
X_treino,X_teste,y_treino,y_teste = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting the Regression model


# Predicting Results
y_pred = regressor.predict()

# Visualiazing Regression Results
plt.figure(figsize = (10,7))

plt.scatter(X, y , color ="red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



# Visualiazing Regression Results in Higher Resolution
X_grid = np.arrange(range(min(X), max(X), 0.1))
X_grid = np.reshape(len(X_grid), 1)

plt.figure(figsize = (10,7))

plt.scatter(X, y , color ="red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()