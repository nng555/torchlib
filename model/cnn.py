import torchlib

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from collections import OrderedDict

class cnn(nn.Module):
   def __init__(self, config):
      super(cnn, self).__init__()
      self.convLayers = []
      self.fcLayers = []
      with open(config, 'rb') as cfile:
         line = cfile.readline()
         while(line != ''):
            subLayers = OrderedDict()
            while(line != '\n'):
               split = line.strip().split()
               lname = split[0]
               params = [int(param) for param in split[1:]]
               if lname == 'fc':
                  subLayers['fc'] = nn.Linear(params[0], params[1])
               elif lname == 'conv2d':
                  subLayers['conv2d'] = nn.Conv2d(
                     params[0],
                     params[1],
                     kernel_size = params[2],
                     padding = params[3]
                  )
               elif lname == 'bnorm':
                  subLayers['batchNorm'] = nn.BatchNorm2d(params[0])
               elif lname == 'relu':
                  subLayers['relu'] = nn.ReLU()
               elif lname == 'pool':
                  subLayers['pool'] = nn.MaxPool2d(params[0])
               line = cfile.readline()
            if 'fc' in subLayers:
               self.fcLayers.append(subLayers['fc'])
            else:
               self.convLayers.append(nn.Sequential(subLayers))
            line = cfile.readline()
      self.convLayers = nn.ModuleList(self.convLayers)
      self.fcLayers = nn.ModuleList(self.fcLayers)

   def forward(self, x):
      out = x
      for convLayer in self.convLayers:
         out = convLayer(out)
      out = out.view(out.size(0), -1)
      for fcLayer in self.fcLayers:
         out = fcLayer(out)
      return out






