#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:20:30 2018

@author: esraa_aldreabi
"""
# the dataset contains information about customers who subsucrib to the mall card. The last colum is spending score which compute points for each customer based on spending, income,num of visit the score is between(1-100).
# problem :segment customer based on annual income and spending score. Scince we don't know how many segments are there this is a cluster problem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3, 4]].values #all raws and col 3 4


# using the Elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

# results: number of cluster is 5

# Applying k-means to mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans= kmeans.fit_predict(X)# fit_predict method that returns for each observation which cluster it's belong to

# visualising the clusters 

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Carefull' )# high income but don't spend much money 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard' )#avg income and avg spending score
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target' )#high income and high spending (target customer)
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'black', label = 'Careless' )#low income and high spending
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'pink', label = 'Sensible' )#low income and low spending

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids' )
plt.title('cluster of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()