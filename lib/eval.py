import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progressbar import *
from torchsample.transforms import *
from torchvision import models

import argparse
import numpy as np

from torchlib.model.cnn import cnn
from torchlib.model.timeres import timeres
from torchlib.model.embryoNet import embryoNet
from torchlib.model.lateFusion import lateFusion
from torchlib.data.embryoDataset import embryoDataset

def eval(conf):
   val_dataset = embryoDataset(
         conf['eval'],
         conf['in_ch'],
         transform=transforms.ToTensor(),
         fusion=('Fusion' in conf['conf_file'])
   )
   val_loader = DataLoader(dataset=val_dataset,
         batch_size=conf['batch_size'],
         shuffle=False,
         num_workers=8)

   if conf['conf_file'] == 'resnet50':
      model = embryoNet(6, False, False, conf['in_ch'], 50)
   elif conf['conf_file'] == 'resnet18':
      model = embryoNet(6, False, False, conf['in_ch'], 18)
   elif conf['conf_file'] == 'lateFusion50':
      model = lateFusion(6, False, False, conf['in_ch'], 50, conf['num_conv'])
   elif conf['conf_file'] == 'lateFusion18':
      model = lateFusion(6, False, False, conf['in_ch'], 18, conf['num_conv'])
   else:
      model = cnn('/home/nathan/torchlib/config/' + conf['conf_file'])

   model_dir = (
         '/data2/nathan/embryo/model/' + conf['conf_file'] + '/' +
         str(conf['lr']) + 'lr_' + str(conf['in_ch']) + 'inch_' + str(conf['optim']) + '/'
   )
   model.cuda()
   model = nn.DataParallel(model)

   model.load_state_dict(torch.load(model_dir + str(conf['nb_epoch']) + '.model'))

   probs_raw = []
   model.eval()
   for frames, times, labels, trans in val_loader:
      frames = [Variable(frame.cuda()) for frame in frames]
      labels = Variable(labels.cuda())
      trans = Variable(trans.cuda()).view(-1, 1).float()

      times = [Variable(time.cuda()).view(-1, 1).float() for time in times]
      outputs = model(frames, times)
      probs_raw.append(outputs.data)

      del frames, times, labels, trans, outputs

   probs_raw = np.asarray(probs_raw)
   probs = []
   for i in range(len(probs_raw)):
          probs.extend(probs_raw[i].cpu().numpy().tolist())
   np.save(open(model_dir + 'probs' + str(conf['nb_epoch']) + conf['eval'] + '.npy', 'wb'), probs)

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
   parser.add_argument('-i', action='store', dest='in_ch', type=int,
         help='the number of channel inputs', default=3)
   parser.add_argument('-c', action='store', dest='num_conv', type=int,
         help='the number of convolution channels', default= 128)
   parser.add_argument('-o', action='store', dest='optim', type=str,
         help='the optimizer to use', default='SGD')
   parser.add_argument('-e', action='store', dest='eval', type=str, default='val',
         help='the data split to evaluate on')
   args = vars(parser.parse_args())

   eval(args)

