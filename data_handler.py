import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from time import time
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import siec
from siec import device

train_set = datasets.ImageFolder(root='./data/zajads_sample',
                                 transform=transforms.Compose([
                                     transforms.Grayscale(),
                                     transforms.ToTensor()
                                 ]))


def imshow(data):
    for img, _ in data:
        plt.imshow(img[0].numpy())
        plt.show()


def imshow_many(imgs):
    n = len(imgs)
    if n < 20:
        cols = 5
        rows = int((n + 4) / 5)
        fig = plt.figure(figsize=(cols * 4, rows * 4))
    else:
        cols = 10
        rows = int((n + 9) / 10)
        fig = plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(n):
        sub = fig.add_subplot(rows, cols, i + 1)
        sub.imshow(imgs[i][0][0].numpy(), interpolation='nearest')
    plt.show()


imshow_many([train_set[i] for i in range(10)])
plt.show()

#%%
def extractFeatures(dataset, model, num_features=18):
    n = len(dataset)
    out = np.zeros((n, num_features))
    with torch.no_grad():
        for (i,(x,_)) in enumerate(dataset):
            if x.shape[-2] < 8 or x.shape[-1] < 8 :
                width = max(8, x.shape[-2])
                height = max(8, x.shape[-1])
                padded = torch.zeros(1,1,width,height)
                padded[0,0,:x.shape[-2], :x.shape[-1]] = x
                x = padded
                x.to(device)
            out[i][:num_features-2] = model.forward_features(x.reshape((1, 1, x.shape[-2], x.shape[-1]))).mean(dim=3).mean(dim=2)[0].cpu().numpy()
            out[i,-2] = x.shape[-2]
            out[i,-1] = x.shape[-1]
    return out,np.arange(n).astype(np.int)


def plotValues(y):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(np.arange(y.shape[0]),y.detach().numpy())
    plt.show()

#%%
model = siec.get_model()

#%%

out,indexes = extractFeatures(train_set, model)





