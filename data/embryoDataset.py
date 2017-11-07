import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image
import cPickle as pkl
import numpy as np

class embryoDataset(data.Dataset):
   def __init__(self, dset, root_dir='/data2/nathan/embryo/', transform=None):
      self.root_dir = root_dir
      self.dset = dset
      self.transform = transform
      self.feat = np.load(root_dir + dset + '/feat.npy')
      self.label = np.load(root_dir + dset + '/labelOrdinal.npy')

   def __getitem__(self, idx):
      frame = self.feat[idx][0]
      time = self.feat[idx][1]
      label = self.label[idx]

      if self.transform is not None:
         frame = self.transform(Image.fromarray(frame))

      return frame, time, label

   def __len__(self):
      return len(self.feat)

