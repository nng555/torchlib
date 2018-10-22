'''
A basic network for embryo stage prediction. Modifies a resnet
to output the requisite number of classes to predict
'''

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import sys
import numpy as np

class embryoNet(nn.Module):

   '''
   Initialize params

   stages       - number of stages to predict
   time         - whether to append time as a feature
   transition   - whether to append transition as a feature
   in_channels  - number of input frames concatenated together
   size         - size of resnet
   '''

   def __init__(self, stages, time, transition, in_channels, size):
      super(embryoNet, self).__init__()

      self.time = time
      self.trans = transition

      # set up base resnet
      if (size == 18):
         models.resnet.model_urls['resnet18'] = 'http://download.pytorch.org/models/resnet18-5c106cde.pth'
         self.resnet = models.resnet18(pretrained=True)

      elif (size == 50):
         models.resnet.model_urls['resnet50'] = 'http://download.pytorch.org/models/resnet50-19c8e357.pth'
         self.resnet = models.resnet50(pretrained=True)

      # set weights for new channel
      c1_old_weights = self.resnet.state_dict()["conv1.weight"].numpy()[:,1,:,:]
      in_cat = [c1_old_weights] * (in_channels)
      c1_new_weights = np.concatenate(tuple(in_cat), axis=1)
      self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False)
      self.resnet.state_dict()["conv1.weight"] = torch.from_numpy(c1_new_weights)

      # output transition regularizer/input time if necessary
      num_ftrs = self.resnet.fc.in_features
      self.resnet.fc = nn.Sequential()
      self.fc = nn.Linear(num_ftrs+self.time, stages+self.trans)

   def forward(self, x, time):
      if (len(x) != 1):
         x = torch.cat(x, dim=1)
      else:
         x = x[0]
      time = time[len(time)/2]

      out = self.resnet(x)
      if(self.time):
         out = self.fc(torch.cat((out, time), dim=1))
      else:
         out = self.fc(out)
      return out
