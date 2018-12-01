##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##
dataset = pd.read_csv("50_Startups.csv")
##
# -------------------------- Separando features e variável dependente --------------------------
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

##
# -------------------------- Cuidando de variáveis categóricas ---------------------------

# Label Encoder transforma os valores da variável categórica em números
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
##
# OneHotEncoder divide a coluna da variável categórica em outras colunas de acordo com o número de categorias
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
##
# -------------------------- Evitando a Dummy Variable Trap --------------------------
X = X[:, 1:]  # Remove a primeira coluna da variável categórica

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
import statsmodels.formula.api as sm
##
# Adicionando a constante para usar o stats model
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
##
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
##
# -------------------------- Backward Elimination - Otimizando o Modelo  --------------------------
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

##
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

##
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
##
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
