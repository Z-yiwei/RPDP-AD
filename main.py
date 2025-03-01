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

from torch.utils.data import DataLoader
from torchvision.utils import save_image


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

        data = np.load("./npy_data/Train_data/" + num_id + ".npy", allow_pickle=True)
        
        #data = Normalization(data)
        #data = data.astype(float)       
    
        
        train_data[flag, :] = data
        train_lab[flag] = 0

        flag = flag + 1

    return train_data, train_lab

Train_data, Train_label = Get_train_data()


def Get_test_data_1(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("npy_data/Test_data/1/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


def Get_test_data_2(test_num):
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


def Get_test_data_3(test_num):
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


def Get_test_data_4(test_num):
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


def Get_test_data_5(test_num):
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

def Get_test_data_6(test_num):
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
        data = np.load("./npy_data/Test_data/6/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab

def Get_test_data_7(test_num):
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
        data = np.load("./npy_data/Test_data/7/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


from deepod.models.tabular import SLAD
from deepod.models.tabular import RDP
# from ablation_study import RDP
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics

clf = RDP()
clf.fit(Train_data, y=None)

Test_data, Test_label = Get_test_data_1(test_num=1400)
scores = clf.decision_function(Test_data)


bins = np.linspace(0, 1, 101)

normal_scores = [scores[i] for i in range(len(scores)) if Test_label[i] == 0]
anomaly_scores = [scores[i] for i in range(len(scores)) if Test_label[i] == 1]

normal_hist, _ = np.histogram(normal_scores, bins=bins)
anomaly_hist, _ = np.histogram(anomaly_scores, bins=bins)

plt.rcParams['font.family'] = 'Times New Roman'

plt.bar(bins[:-1], normal_hist, width=0.01, alpha=0.5, color='blue', label='Normal')
plt.bar(bins[:-1], anomaly_hist, width=0.01, alpha=0.5, color='red', label='Anomaly')

plt.legend()

plt.xlabel('Anomaly Score')
plt.ylabel('Number of Samples')

plt.savefig('anomaly_score_distribution.svg', format='svg')

plt.show()


