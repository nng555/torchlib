import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image
import cPickle as pkl
import numpy as np
import cv2

class embryoDataset(data.Dataset):
   def __init__(self, dset, root_dir='/data2/embryo/', transform=None):
      self.root_dir = root_dir
      self.dset = dset
      self.transform = transform
      self.feat = np.load(root_dir + dset + '/featRaw.npy')
      self.label = np.load(root_dir + dset + '/labelOrdinal.npy')

   def __getitem__(self, idx):
      if (idx - 1) > 0 and self.feat[idx-1][1] < self.feat[idx][1]:
         frame0 = self.feat[idx-1][0]
      else:
         frame0 = self.feat[idx][0]
      frame1 = self.feat[idx][0]
      if (idx + 1) < len(self.feat) and self.feat[idx][1] < self.feat[idx+1][1]:
         frame2 = self.feat[idx+1][0]
      else:
         frame2 = self.feat[idx][0]
      gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
      gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
      frame = np.dstack([gray0, gray1, gray2])
      time = self.feat[idx][1]
      label = self.label[idx]

      if self.transform is not None:
         frame = self.transform(Image.fromarray(frame))

      return frame, time, label

   def __len__(self):
      return len(self.feat)

