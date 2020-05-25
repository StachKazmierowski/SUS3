import data_handler_norm
from data_handler_norm import train_set
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering



X = np.zeros((len(train_set), data_handler_norm.IMG_SIZE**2))

for i in range(len(train_set)):
    x = train_set[i][0]
    X[i] = x.reshape(-1).numpy()

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

printToHtml(labeled[:2000])




