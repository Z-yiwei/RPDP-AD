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

test_num = 1400

def Get_test_data_1():
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1
        
    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/1/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1        
        
    return test_data, test_lab

test_num = 1400

def Get_test_data_2():
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1
        
    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/2/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1        
        
    return test_data, test_lab

test_num = 1400

def Get_test_data_3():
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1
        
    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/3/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1        
        
    return test_data, test_lab

test_num = 1400

def Get_test_data_4():
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1
        
    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/4/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1        
        
    return test_data, test_lab

test_num = 1400

def Get_test_data_5():
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1
        
    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/5/" + num_id + ".npy", allow_pickle=True)     
        
        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1        
        
    return test_data, test_lab

from deepod.models.tabular import RDP
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics

clf = RDP()
clf.fit(Train_data, y=None)
