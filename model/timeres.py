import torch
import torch.nn as nn

class timeres(nn.Module):

   def __init__(self, resnet, num_ftrs):
      super(timeres, self).__init__()
      self.resnet = resnet
      self.fc = nn.Linear(num_ftrs + 1, 7)

   def forward(self, x, time):
      res = self.resnet(x)
      res = torch.cat((res, time), 1)
      res = self.fc(res)
      return res


