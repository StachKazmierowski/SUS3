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


def accuracy(model, testloader, epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy", 100 * correct / total)
        return 100 * correct / total

def get_model(trained=False):
    model = Net(device=device)
    if trained :
        model.load_state_dict(torch.load('./mnist_net'))
        model.to(device)
        print("loaded")
        return model
    else :
        return model




