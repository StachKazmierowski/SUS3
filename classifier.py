from scipy.cluster.hierarchy import dendrogram

import numpy as np
from matplotlib import pyplot as plt

from data_handler import out as X
from data_handler  import indexes
from data_handler import train_set
from sklearn.cluster import KMeans


#%%
from sklearn.metrics import silhouette_score
def silhouette_optimizer(k_max, X):
    k_max = k_max
    number_of_steps = 10
    step = int(k_max / number_of_steps)
    k = step
    best_value = 0
    best_value_id = 0
    while (step >= 1):
        while (k < k_max):
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            tmp_val = silhouette_score(X, labels, metric='euclidean')
            if(tmp_val>best_value):
                best_value = tmp_val
                best_value_id = k
            k += step
        step = int(step/2)
        best_value = 0
        k = int(max(step, (best_value_id - (number_of_steps * step / 2))))
        k_max = int(min((best_value_id + (number_of_steps * step / 2)), k_max))
    return best_value_id

#%%
def plot_optimizer_variance(k_max, step, X):
    best_ids = []
    ranges = []
    k = 100
    while(k < k_max):
        best_ids.append(silhouette_optimizer(k, X))
        ranges.append(k)
        k += step
    plt.plot(ranges, best_ids)
    plt.title("checking maximizing silhouette score variance")
    plt.xlabel("max number of clusters")
    plt.show()
plot_optimizer_variance(200, 50, X)
#%%
def silhouette_score_plot(k_max, step, X):
    k = step
    sils = []
    args = []
    while (k < k_max):
        kmeans = KMeans(n_clusters=k).fit(X)
        labels = kmeans.labels_
        sils.append(silhouette_score(X, labels, metric='euclidean'))
        args.append(k)
        k += step
    plt.plot(args, sils)
    plt.title("silhouette score - looking for highest")
    plt.xlabel("number of clusters")
    plt.show()
    print(args)

silhouette_score_plot(500, 20, X)
#%%
def elbow_point_plot(k_max, step, X):
    sse = []
    args = []
    k = step
    while(k < k_max + 1):
        kmeans = KMeans(n_clusters=k).fit(X)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(X)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(X)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (X[i, 0] - curr_center[0]) ** 2 + (X[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
        args.append(k)
        k += step
    plt.plot(args, sse)
    plt.title("elbow method - looking for elbow point")
    plt.xlabel("number of clusters")
    plt.show()
elbow_point_plot(500, 40, X)

