##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##
# -------------------------- Cuidando de dados vazios --------------------------
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])

##
# -------------------------- Cuidando de variáveis categóricas ---------------------------

# Label Encoder transforma os valores da variável categórica em números
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# Variável Dependente
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
##
# OneHotEncoder divide a coluna da variável categórica em outras colunas de acordo com o número de categorias
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

##
# -------------------------- Aplicando Feature Scaling  --------------------------
'''from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_treino = scaler_X.fit_transform(X_treino)
X_teste = scaler_X.transform(X_teste)'''
