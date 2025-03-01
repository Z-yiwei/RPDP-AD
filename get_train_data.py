import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor

from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

train_num = 600

def Get_train_data():
    train_data = np.zeros((train_num, 201), dtype=np.float32)
    train_lab = np.zeros((train_num), dtype=np.float32)

    flag = 0

    for id in range(train_num):
        num_id = str(id)

        data = np.load("./npy_data/train_data/" + num_id + ".npy", allow_pickle=True)
        
        #data = Normalization(data)
        #data = data.astype(float)       
    
        
        train_data[flag, :] = data
        train_lab[flag] = 0

        flag = flag + 1

    return train_data, train_lab

Train_data, Train_label = Get_train_data()
