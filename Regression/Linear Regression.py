import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##
dataset = pd.read_csv("Salary.csv")
##
# -------------------------- Separando features e variável dependente --------------------------
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

##
# -------------------------- Separando os dados em Treino e Teste --------------------------
from sklearn.model_selection import train_test_split
X_treino,X_teste,y_treino,y_teste = train_test_split(X, y, test_size=0.2, random_state=0)

##
# -------------------------- Importando e criando o Regressor --------------------------
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treino, y_treino)

##
# -------------------------- Prevendo os resultados em Teste --------------------------
y_pred = regressor.predict(X_teste)

##
# -------------------------- Visualizando os resultados em Treino --------------------------
plt.scatter(X_treino, y_treino, c="red",)
plt.plot(X_treino, regressor.predict(X_treino), c="blue")
plt.title('Salario vs Anos de Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')

##
# -------------------------- Visualizando os resultados em Teste --------------------------
plt.scatter(X_teste, y_teste, c="red",)
plt.plot(X_treino, regressor.predict(X_treino), c="blue")
plt.title('Salario vs Anos de Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')