import data_handler_norm
from data_handler_norm import train_set
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from matplotlib import pyplot as plt


X = np.zeros((len(train_set), data_handler_norm.IMG_SIZE**2))

for i in range(len(train_set)):
    x = train_set[i][0]
    X[i] = x.reshape(-1).numpy()

#%%

from sklearn import metrics

def train_model(n=100):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = MiniBatchKMeans(n_clusters=n)
    model = model.fit(X)
    return  model

def calculate_metrics(model):
    print('Number of clusters is {}'.format(model.n_clusters))
    print('Inertia : {}'.format(model.inertia_))
    print('silhuette : {}'.format(metrics.silhouette_score(X, model.labels_, metric = 'euclidean')))


from sklearn.cluster import KMeans


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points):
    sse = []
    for k in [80,100,150,200,300,400]:
        kmeans = train_model(k)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        print("K=", k)
        calculate_metrics(kmeans)
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.sum((points[i] - curr_center) ** 2)

        sse.append(curr_sse)
    return sse

#%%
# sse = calculate_WSS(X)
# plt.plot(sse)
#
#%%
n=100
model = train_model(n)
labels = model.labels_
calculate_metrics(model)
#%%
distances = model.transform(X)
ind_distances = distances[np.arange(len(distances)), labels]
mean_dist = np.zeros(model.n_clusters)
cl_size = np.zeros(model.n_clusters)
for i in range(X.shape[0]) :
    cl = model.labels_[i]
    cl_size[cl] += 1
    mean_dist[cl] += ind_distances[i]
mean_dist = mean_dist/cl_size
argsorted = np.argsort(mean_dist)
scores = np.zeros(model.n_clusters)
scores[argsorted] = np.arange(model.n_clusters)
#%%


import os

def filenames(indices=[], basename=False):
    if indices:
        # grab specific indices
        if basename:
            return [os.path.basename(train_set.imgs[i][0]) for i in indices]
        else:
            return [train_set.imgs[i][0] for i in indices]
    else:
        if basename:
            return [os.path.basename(x[0]) for x in train_set.imgs]
        else:
            return [x[0] for x in train_set.imgs]


samples = filenames()
labeled = sorted([x for x in zip(samples, labels)], key=lambda x: scores[x[1]])
labeled[-5:]
#%%

score = metrics.silhouette_score(X, model.labels_, metric = 'euclidean')


# %%

HTML_FILE_BEGINING = """
<!doctype html>

<html lang="en">
<body>
"""

HTML_FILE_END = """
</body>
</html>
"""

SEPARATOR = """..."""

def printToHtml(data, filename='out.html'):
    with open('out.html', 'w+') as f:
        f.write(HTML_FILE_BEGINING)
        _, last_group = data[0]
        for fname, group in data:
            if group != last_group:
                f.write("<HR>")
                last_group = group
            f.write("<img src=\"{0}\" alt=\"{1}\">{2}".format(fname, fname, SEPARATOR))

        f.write(HTML_FILE_END)


# %%

printToHtml(labeled[-2000:])

#%%

# emnist
from emnist import extract_training_samples
images, labels = extract_training_samples('balanced')

