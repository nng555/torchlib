import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.legacy.nn as legacy
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

@profile
def load_test():
   train_dataset = embryoDataset(
         'train',
         transform=transforms.Compose([
            transforms.ToTensor(),
            RandomRotate(360),
         ]))
   train_loader = DataLoader(dataset=train_dataset,
         batch_size=90,
         shuffle=True,
         num_workers=8)

   num_batches = len(train_dataset)/90
   widgets = [Percentage(), ' ', Bar(marker='=',left='[',right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
   pbar = ProgressBar(widgets=widgets, maxval=num_batches)
   pbar.start()
   for i, (frames, times, labels, trans) in enumerate(train_loader):
      frames = Variable(frames.cuda(), volatile=True)
      labels = Variable(labels.cuda(), volatile=True)
      times = Variable(times.cuda(), volatile=True).view(-1, 1).float()
      trans = Variable(trans.cuda(), volatile=True).view(-1, 1).float()
      pbar.update(i)
      del frames, times, labels, trans
   pbar.finish()

   widgets = [Percentage(), ' ', Bar(marker='=',left='[',right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
   pbar = ProgressBar(widgets=widgets, maxval=num_batches)
   pbar.start()
   for i, (frames, times, labels, trans) in enumerate(train_loader):
      frames = Variable(frames.cuda(), volatile=True)
      labels = Variable(labels.cuda(), volatile=True)
      times = Variable(times.cuda(), volatile=True).view(-1, 1).float()
      trans = Variable(trans.cuda(), volatile=True).view(-1, 1).float()
      pbar.update(i)
      del frames, times, labels, trans
   pbar.finish()

if __name__ == "__main__":
   load_test()
