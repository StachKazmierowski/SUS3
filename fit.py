import data_handler_norm
from data_handler_norm import train_set
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from matplotlib import pyplot as plt
from sklearn import metrics
import os

X = np.zeros((len(train_set), data_handler_norm.IMG_SIZE**2))

for i in range(len(train_set)):
    x = train_set[i][0]
    X[i] = x.reshape(-1).numpy()


#%%



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

def train_model(n=100):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = MiniBatchKMeans(n_clusters=n)
    model = model.fit(X)
    return  model

def calculate_metrics(X, model):
    ine = model.inertia_
    sil = metrics.silhouette_score(X, model.labels_, metric = 'euclidean')
    return ine, sil, model.score(X)


def grade_clusters(X, labels, model):
    distances = model.transform(X)
    ind_distances = distances[np.arange(len(distances)), labels]
    mean_dist = np.zeros(model.n_clusters)
    max_dist = np.zeros((model.n_clusters, 5))
    cl_size = np.zeros(model.n_clusters)
    for i in range(X.shape[0]):
        cl = model.labels_[i]
        cl_size[cl] += 1
        dist = ind_distances[i]
        mean_dist[cl] += dist
        dists = max_dist[cl]
        if dist > dists.min() :
            max_dist[cl,np.argmin(dists)] = dist
    mean_dist = mean_dist / cl_size
    mean_max_dist = max_dist.mean(axis=1)
    argsorted = np.argsort(mean_max_dist)
    scores = np.zeros(model.n_clusters)
    scores[argsorted] = np.arange(model.n_clusters)
    return scores


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
    with open(filename, 'w+') as f:
        f.write(HTML_FILE_BEGINING)
        _, last_group = data[0]
        for fname, group in data:
            if group != last_group:
                f.write("<HR>")
                last_group = group
            f.write("<img src=\"{0}\" alt=\"{1}\">{2}".format(fname, fname, SEPARATOR))

        f.write(HTML_FILE_END)





#%%


def get_data(n):
    model = train_model(n)
    samples = filenames()
    labels = model.labels_
    scores = grade_clusters(X, labels, model)
    labeled = sorted([x for x in zip(samples, labels)], key=lambda x: scores[x[1]])

    printToHtml(labeled[-2000:], "tests/worst{0}.html".format(n))
    printToHtml(labeled[:2000], "tests/best{0}.html".format(n))
    return calculate_metrics(X, model)

#%%
inertias = []
sils = []
scs = []

eny =  [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

for n in eny:
    ine, sil, sc = get_data(n)
    inertias.append(ine)
    sils.append(sil)
    scs.append(sc)

#%%
plt.figure()
plt.plot(eny, inertias)
plt.title("inertia")
plt.figure()
plt.plot(eny, sils)
plt.title("siluet")
plt.figure()
plt.plot(eny, scs)
plt.title("metrics.score")
