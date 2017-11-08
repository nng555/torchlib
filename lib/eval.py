import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progressbar import *
from torchsample.transforms import *
from torchvision import models

import argparse
import numpy as np

from torchlib.model.cnn import cnn
from torchlib.data.embryoDataset import embryoDataset

def eval(conf):
   val_dataset = embryoDataset(
      'val',
      transform=transforms.Compose([
         #transforms.TenCrop(200),
         #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
         transforms.ToTensor()
      ])
   )
   val_loader = DataLoader(dataset=val_dataset,
         batch_size=conf['batch_size'],
         shuffle=False)
   if conf['conf_file'] == 'resnet50':
      models.resnet.model_urls['resnet50'] = 'http://download.pytorch.org/models/resnet50-19c8e357.pth'
      model = models.resnet50(pretrained=True)
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs, 6)

   model.cuda()
   model = nn.DataParallel(model)
   model.load_state_dict(torch.load('/home/ubuntu/model/resnet' + str(conf['lr']) + str(conf['nb_epoch']) + '.model'))

   model.eval()
   preds = []
   for frames, times, labels in val_loader:
      #bs, ncrops, c, h, w = frames.size()
      frames = Variable(frames.cuda())
      times = Variable(times.cuda())
      labels = Variable(labels.cuda())
      outputs = model(frames)
      _, prediction = torch.max(outputs.data, 1)
      preds.append(prediction)
      #result = model(frames.view(-1, c, h, w))
      #result_avg = result.view(bs, ncrops, -1).mean(1)

   preds = np.asarray(preds)
   np.save(open('preds.npy', 'wb'), preds)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-d', action='store', dest='dataset', type=str,
         help='the dataset to train on')
   parser.add_argument('-l', action='store', dest='loss', type=str,
         help='the loss function')
   parser.add_argument('-lr', action='store', dest='lr', type=float,
         help='the learning rate')
   parser.add_argument('-n', action='store', dest='nb_epoch', type=int, default=300,
         help='the max number of epochs to train for')
   parser.add_argument('-b', action='store', dest='batch_size', type=int, default=256,
         help='the batch size')
   parser.add_argument('-f', action='store', dest='conf_file', type=str,
         help='the architecture config file')
   args = vars(parser.parse_args())

   eval(args)

