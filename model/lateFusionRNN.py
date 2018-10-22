import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models
import sys
import numpy as np

class lateFusionRNN(nn.Module):

   def __init__(self):
      super(lateFusionRNN, self).__init__()


