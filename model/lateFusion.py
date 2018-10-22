import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models
import sys
import numpy as np

class lateFusion(nn.Module):

   def __init__(self, stages, time, transition, in_channels, size, num_conv):
      super(lateFusion, self).__init__()

      self.time = time
      self.trans = transition

      # set up base resnet
      if (size == 50):
         models.resnet.model_urls['resnet50'] = 'http://download.pytorch.org/models/resnet50-19c8e357.pth'
         self.resnet = models.resnet50(pretrained=True)

      elif (size == 18):
         models.resnet.model_urls['resnet18'] = 'http://download.pytorch.org/models/resnet18-5c106cde.pth'
         self.resnet = models.resnet18(pretrained=True)

      # set weights for new channel
      c1_old_weights = self.resnet.state_dict()["conv1.weight"].numpy()[:,1,:,:]
      in_cat = [c1_old_weights] * (in_channels)
      c1_new_weights = np.concatenate(tuple(in_cat), axis=1)
      self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
            stride=2, padding=3, bias=False)
      self.resnet.state_dict()["conv1.weight"] = torch.from_numpy(c1_new_weights)

      # output transition regularizer/input time if necessary
      num_ftrs = self.resnet.fc.in_features
      print(num_ftrs)
      self.resnet.fc = nn.Sequential()
      if(self.time):
         if (size == 50):
            self.conv = nn.Conv2d(1, num_conv, (3, 2049))
         elif (size == 18):
            self.conv = nn.Conv2d(1, num_conv, (3, 513))
      else:
         if (size == 50):
            self.conv = nn.Conv2d(1, num_conv, (3, 2048))
         elif (size == 18):
            self.conv = nn.Conv2d(1, num_conv, (3, 512))
      self.fc = nn.Linear(num_conv * (in_channels - 3 + 1), stages+self.trans)
      self.dropout = nn.Dropout()

   # x is in_ch x N x 1 x 224 x 224
   # time is in_ch x N
   def forward(self, x, time):
      reprs = []
      for i in range(len(x)):
         out = self.resnet(x[i]) # N x 2048
         if(self.time):
            out = torch.cat((out, time[i]), dim=1) # N x 2049
            out = torch.unsqueeze(out, dim=1)
         else:
            out = torch.unsqueeze(out, dim=1)
         reprs.append(out)
      out = torch.cat(reprs, dim = 1) # N x in_ch x 2049
      out = torch.unsqueeze(out, dim=1) # N x 1 x in_ch x 2049
      out = F.relu(torch.squeeze(self.conv(out), 3)) # N x num_conv x in_ch
      #out = F.max_pool1d(out, )
      out = out.view(x[0].size(0), -1)
      out = self.fc(self.dropout(out))
      return out
