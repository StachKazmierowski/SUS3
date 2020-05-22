import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from time import time
from torchvision import datasets, transforms
from torch.autograd import Variable