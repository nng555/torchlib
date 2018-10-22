import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progressbar import *
from torchsample.transforms import *
from torchvision import models
from itertools import ifilter

import argparse
import numpy as np

from torchlib.model.cnn import cnn
from torchlib.model.timeres import timeres
from torchlib.model.embryoNet import embryoNet
from torchlib.data.embryoDataset import embryoDataset
from torchlib.model.lateFusion import lateFusion

val_dataset = embryoDataset('val', 5, transform=transforms.Compose([transforms.ToTensor(), RandomRotate(360)]),fusion=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=8)
for frames, times, labels, trans in val_loader:
   break
times = [Variable(torch.unsqueeze(time, 1).float()) for time in times]
frames = [Variable(frame) for frame in frames]
model = lateFusion(6, True, False, 5)
