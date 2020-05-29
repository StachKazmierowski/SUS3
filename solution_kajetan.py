import sys

import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
import ntpath
from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast, adjust_brightness



if (len(sys.argv) != 2) :
    print("Error, wrong arguments")
    print("Usage: {0} <filename>".format(sys.argv[0]))
    print("    filename - name of file with list of image paths")
    exit(1)

filename = sys.argv[1]
lines = []
with open(filename) as f:
    lines = f.readlines()

lines = [x[:-1] for x in lines]



IMG_SIZE = 50

def pad(img):
    img = np.array(img)
    x, y= img.shape
    size = IMG_SIZE
    out = np.ones((size, size))*255
    if x > y :
        dif = x - y
        left = int(dif / 2)
        top = 0
    elif y > x :
        dif = y - x
        top = int(dif/2)
        left = 0
    else :
        top = 0
        left = 0
    out[top:top+x, left:left+y] = img
    return out



def resize(img):
    x, y = img.size
    if x > y :
        rate = IMG_SIZE/float(x)
    else :
        rate = IMG_SIZE/float(y)
    return img.resize((int(x*rate), int(y*rate)))


def contrast(img):
    cont = adjust_contrast(img, 1.2)
    return adjust_brightness(cont, 1.2)


X = np.zeros((len(lines), IMG_SIZE**2))


transform= transforms.Compose([transforms.Grayscale(),
                                     contrast,
                                     resize,
                                     pad,
                                     transforms.ToTensor()
                                 ])


for i,image_path in enumerate(lines) :
    img = Image.open(image_path)
    X[i] = transform(img).reshape(-1).numpy()

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

SEPARATOR = """"""

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

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def printToFile(data, filename='out.txt'):
    with open(filename, 'w+') as f:
        _, last_group = data[0]
        for fname, group in data:
            if group != last_group:
                f.write("\n")
                last_group = group
            f.write("{0} ".format(path_leaf(fname)))




def get_data(n):
    model = train_model(n)
    samples = filenames()
    labels = model.labels_
    scores = grade_clusters(X, labels, model)
    labeled = sorted([x for x in zip(samples, labels)], key=lambda x: scores[x[1]])

    printToHtml(labeled[-2000:], "tests/worst{0}.html".format(n))
    printToHtml(labeled[:2000], "tests/best{0}.html".format(n))
    return calculate_metrics(X, model)


inertias = []
sils = []
scs = []

eny =  np.arange(30,130,5)

for n in eny:
    ine, sil, sc = get_data(n)
    inertias.append(ine)
    sils.append(sil)
    scs.append(sc)


inertias = np.array(inertias)
sils = np.array(sils)

# plt.figure()
# plt.plot(eny, sils)
# plt.title("siluet")
# plt.figure()
# plt.plot(eny, scs)
# plt.title("metrics.score")
# plt.figure()
#%%



# plt.plot(eny, inertias)
# plt.title("inertia")

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


guess = np.polyfit(eny, np.log(inertias), 1)
guess_lin = np.array([ np.exp(guess[1]),  np.abs(guess[0]),  np.min(inertias)])
popt, pcov = curve_fit(func, eny, inertias, p0=guess_lin)
# print(popt)


# plt.plot(eny, func(eny, guess_lin[0], guess_lin[1], 0), 'g--',
#          label='lin_fit: a=%5.3f, b=%5.3f, c=%5.3f' % (guess_lin[0], guess_lin[1], 0))
#
# plt.plot(eny, func(eny, *popt), 'r--',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# plt.legend()
# plt.show()


#%%
from kneed import KneeLocator
x = np.arange(1,120,1)
kn = KneeLocator(x, func(x, *popt), curve='convex', direction='decreasing')
knee = kn.knee
# print("knee found ", knee)


model = train_model(knee)
samples = lines
labels = model.labels_
scores = grade_clusters(X, labels, model)
labeled = sorted([x for x in zip(samples, labels)], key=lambda x: scores[x[1]])

printToHtml(labeled[:2000])
printToFile(labeled[:2000])







