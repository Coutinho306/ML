##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##
dataset = pd.read_csv("C:\\Users\\Thiago\\Desktop\\ML\\Datasets\\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

##
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    cluster = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    cluster.fit(X)
    wcss.append(cluster.inertia_)
plt.plot(range(1,11), wcss)
plt.ylabel("WCSS")
plt.xlabel("Number of Clusters")
plt.title("The Elbow Method")
plt.xlim(1, 10)



##
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

##
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="cyan", label="cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="magenta", label="cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="Centroids")
plt.title("Cluster of Clients")
plt.xlabel("Annual Incomme (K$) ")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
