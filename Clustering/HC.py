##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##
dataset = pd.read_csv("C:\\Users\\Thiago\\Desktop\\ML\\Datasets\\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

##
# Using Dendrogram to find the optimal number os Clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

##
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

##
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c="red", label="cluster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c="blue", label="cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c="green", label="cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c="cyan", label="cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c="magenta", label="cluster 5")
plt.title("Cluster of Clients")
plt.xlabel("Annual Incomme (K$) ")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

