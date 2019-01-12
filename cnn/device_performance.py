from collections import OrderedDict
import time
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from operations import *
from genotypes import PRIMITIVES
from model_search import Network
from architect import Architect

C = 16
C_curr = C * 3
stride = 1
affine = False

device = "cpu"
transform_train = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=1)

performances = OrderedDict()


def stems(out=16):
    C_curr = 16 * 3
    stem = nn.Sequential(
        nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr),
        ReLUConvBN(C_curr, out, 1, 1, 0, affine=False))
    return stem


for k, v in OPS.items():
    for s in [16, 32, 64]:
        total_time = 0
        operation = v(C=s, stride=stride, affine=affine).to(device)
        for batch_idx, (inputs, _) in enumerate(trainloader):
            if batch_idx % 50000 == 0:
                print(k, s, batch_idx)
            inputs = inputs.to(device)
            stem = stems(s).to(device)
            inputs = stem(inputs)
            start_time = time.time()
            outputs = operation(inputs)
            end_time = time.time()
            total_time += end_time - start_time
        performances["{0}_{1}".format(k, s)] = {
            "total_second": total_time,
            "average_second": total_time / len(trainset.train_labels)
        }

print(performances)

a = pd.DataFrame(performances).T
print(a)

a.to_csv("./performance.csv")
