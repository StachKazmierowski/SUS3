from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

import numpy as np
from matplotlib import pyplot as plt

from data_handler import out as X
from data_handler  import indexes
from data_handler import train_set


# setting distance_threshold=0 ensures we compute the full tree.
clustering = AgglomerativeClustering(n_clusters=100).fit(X)

#%%
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#%%
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(n_clusters=100)

model = model.fit(X)
#%%

labels = model.labels_

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

printToHtml(labeled[:1000])
