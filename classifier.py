from scipy.cluster.hierarchy import dendrogram

import numpy as np
from matplotlib import pyplot as plt

from data_handler import out as Xmean_18
from data_handler import out2 as Xmax_18
from data_handler import out3 as Xmean_26
from data_handler import out4 as Xmax_26
from data_handler import out5 as Xmean_10
from data_handler import out6 as Xmax_10
# from data_handler import indexes
from data_handler import train_set
from sklearn.cluster import KMeans

models = [Xmean_10, Xmax_10, Xmean_18, Xmax_18, Xmean_26, Xmax_26]
names = ['Xmean_10', 'Xmax_10', 'Xmean_18', 'Xmax_18', 'Xmean_26', 'Xmax_26']


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
def plot_optimizer_variance(k_max, step, X, name):
    best_ids = []
    ranges = []
    k = 100
    while(k < k_max):
        best_ids.append(silhouette_optimizer(k, X))
        ranges.append(k)
        k += step
    plt.plot(ranges, best_ids)
    plt.title("checking maximizing silhouette score variance " + name)
    plt.xlabel("max number of clusters")
    plt.savefig('./plots/sil_var_' + name + '.png')
    plt.show()
# plot_optimizer_variance(200, 50, Xmean_10, 'lol')
#%%
def silhouette_score_plot(k_max, step, X, name):
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
    plt.title("silhouette score - looking for highest " + name)
    plt.xlabel("number of clusters")
    plt.savefig('./plots/sil_score_' + name + '.png')
    plt.show()
    # print(args)

# silhouette_score_plot(500, 20, Xmean)
# silhouette_score_plot(500, 20, Xmax)
#%%
def elbow_point_plot(k_max, step, X, name):
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
    plt.title("elbow method - looking for elbow point " + name)
    plt.xlabel("number of clusters")
    plt.savefig('./plots/elbow_' + name + '.png')
    plt.show()
# elbow_point_plot(500, 40, Xmean)
# elbow_point_plot(500, 40, Xmax_10, 'Xmax_10')

#%%
# model = KMeans(n_clusters=200)
# model = model.fit(Xmax)
#%%
def model_survey(model, name, kmax, step):
    plot_optimizer_variance(300, 50, model, name)
    silhouette_score_plot(kmax, step, model, name)
    elbow_point_plot(kmax, step, model, name)

for i in range(len(models)):
    model_survey(models[i], names[i], 500, 25)
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
labeled = sorted([x for x in zip(samples, labels)], key=lambda x: x[1])
labeled[:5]

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
printToHtml(labeled[1000:2000])
#%%
