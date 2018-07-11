import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image
import cPickle as pkl
import numpy as np
import cv2

class embryoDataset(data.Dataset):
   def __init__(self, cut, dset, in_channels, root_dir='/data2/nathan/embryo/', transform=None, fusion=False):
      self.root_dir = root_dir
      self.dset = dset
      self.transform = transform
      if cut:
         self.feat = np.load(root_dir + dset + '/featCut.npy')
         self.label = np.load(root_dir + dset + '/labelCutIndex.npy')
      else:
         self.feat = np.load(root_dir + dset + '/featRaw.npy')
         self.label = np.load(root_dir + dset + '/labelRawIndex.npy')
      self.in_channels = in_channels
      self.fusion = fusion

   def __getitem__(self, idx):
      if(self.in_channels != 1):
         mid_c = self.in_channels/2
         frame = [0 for i in range(self.in_channels)]
         frame[mid_c] = self.feat[idx][0]
         times = [0 for i in range(self.in_channels)]
         times[mid_c] = self.feat[idx][1]

         # fill the front half of the array with the last valid frame
         last1 = 0
         for i in range(1, mid_c + 1):
            if idx - i > 0 and self.feat[idx-i][1] < self.feat[idx][1]:
               last1 = i
            frame[mid_c - i] = self.feat[idx-last1][0]
            times[mid_c - i] = self.feat[idx-last1][1]

         # fill the back half of the array with the last valid frame
         last2 = 0
         for i in range(1, mid_c + 1):
            if idx + i < len(self.feat) and self.feat[idx][1] < self.feat[idx+i][1]:
               last2 = i
            frame[mid_c + i] = self.feat[idx+last2][0]
            times[mid_c + i] = self.feat[idx+last2][1]

         if self.transform is not None:
            frame = [self.transform(Image.fromarray(img)).narrow(0, 0, 1) for img in frame]
         #frame = torch.stack(tuple(frame), dim=0)
      else:
         #frame = cv2.cvtColor(self.feat[idx][0], cv2.COLOR_BGR2GRAY)
         #frame = np.reshape(frame, (len(frame), len(frame[0]), 1))
         if self.transform is not None:
            frame = self.transform(Image.fromarray(self.feat[idx][0])).narrow(0, 0, 1)
         frame = [frame]
         times = [self.feat[idx][1]]

      label = self.label[idx]
      transition = False
      if (idx - 1) > 0 and self.label[idx - 1] < self.label[idx]:
         transition = True

      return frame, times, label, transition

   def __len__(self):
      return len(self.feat)

