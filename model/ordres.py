import torch
import torch.nn as nn

class ordres(nn.Module):

   def __init__(self, resnet):
      super(ordres, self).__init__()
      self.resnet = resnet

   def forward(self, x):
      self.res = self.resnet(x)
      self.sp = nn.Softplus()(self.res)
      self.ordsum = [torch.sum(self.sp.narrow(0, i, 6)) for i in range(6)]
      self.sig = [nn.Sigmoid()(val) for val in self.ordsum]
      return torch.cat(self.sig)