import numpy as np  # to handle matrix and data operation
import pandas as pd  # to read csv and handle dataframe

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from time import time
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = "cuda:0"
    print("detected cuda device")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = "cpu"
    torch.set_default_tensor_type(torch.FloatTensor)

BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('./data/train', download=True, train=True, transform=transform)
valset = datasets.MNIST('./data/test', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)


# %%

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear2 = nn.Linear(6 * 6 * 16, 10)

    def forward(self, X):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        X = X.view(-1, 6 * 6 * 16)
        X = self.linear2(X)
        return X

    def forward_features(self, X):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        return X


model = Net(device=device)

