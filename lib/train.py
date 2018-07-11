import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progressbar import *
from torchsample.transforms import *
from torchvision import models
from itertools import ifilter

import argparse
import numpy as np

from torchlib.model.cnn import cnn
from torchlib.model.timeres import timeres
from torchlib.model.embryoNet import embryoNet
from torchlib.data.embryoDataset import embryoDataset
from torchlib.model.lateFusion import lateFusion
from torchlib.model.lateFusionFrozen import lateFusionFrozen

# change learning rate at beginning of each epoch
def adjust_learning_rate(optimizer, decay, epoch, conf):
   lr = conf['lr'] * (0.5 ** (epoch//conf['h_rate']))
   for param_group in optimizer.param_groups:
      param_group['lr'] = lr

# generate the transition loss based on
def trans_loss(outputs, labels, trans, critStage, critTrans, numLabels):
   ostages = outputs.narrow(1, 0, 6)
   _, preds = torch.max(nn.functional.softmax(ostages), 1)
   otrans = outputs.narrow(1, 6, 1)
   loss = 0.75*critStage(ostages, labels) + 0.25*critTrans(otrans, trans)
   return loss, preds

def simple_loss(outputs, labels, critStage):
   loss = critStage(outputs, labels)
   _, preds = torch.max(nn.functional.softmax(outputs), 1)
   return loss, preds

def load_datasets(conf):
   print("Loading datasets...")
   train_dataset = embryoDataset(
         ('cut' in conf['dataset']),
         'train',
         conf['in_ch'],
         transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(360),
            transforms.RandomCrop(200),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
         ]),
         fusion=('Fusion' in conf['conf_file'])
   )
   val_dataset = embryoDataset(
         ('cut' in conf['dataset']),
         'val',
         conf['in_ch'],
         transform=transforms.ToTensor(),
         fusion=('Fusion' in conf['conf_file'])
   )

   train_loader = DataLoader(dataset=train_dataset,
         batch_size=conf['batch_size'],
         shuffle=True,
         num_workers=8)
   val_loader = DataLoader(dataset=val_dataset,
         batch_size=conf['batch_size'],
         shuffle=False,
         num_workers=8)
   print("Datasets loaded")
   return train_dataset, val_dataset, train_loader, val_loader

def load_vggnet():
   print("Loading Model...")
   models.vgg.model_urls['vgg19_bn'] = 'http://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
   model = models.vgg19_bn(pretrained=False, num_classes=7)
   model_dict = model_zoo.load_url(models.vgg.model_urls['vgg19_bn'])
   model_dict['classifier.6.weight'] = model_dict['classifier.6.weight'].narrow(0,0,7)
   model_dict['classifier.6.bias'] = model_dict['classifier.6.bias'].narrow(0,0,7)
   model.load_state_dict(model_dict)
   model.cuda()
   model = nn.DataParallel(model)
   print("Model loaded")
   return model

def train_epoch(model, train_loader, optimizer, critStage, critTrans, num_batches, conf):
   widgets = [Percentage(), ' ', Bar(marker='=',left='[',right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
   pbar = ProgressBar(widgets=widgets, maxval=num_batches)
   pbar.start()
   loss_acm = 0
   acc_acm = 0
   for i, (frames, times, labels, trans) in enumerate(train_loader):
      frames = [Variable(frame.cuda()) for frame in frames]
      labels = Variable(labels.cuda())
      trans = Variable(trans.cuda()).view(-1, 1).float()
      optimizer.zero_grad()

      times = [Variable(time.cuda()).view(-1, 1).float() for time in times]
      outputs = model(frames, times)

      if conf['conf_file'] == 'timerestrans':
         loss, preds = trans_loss(outputs, labels, trans, critStage, critTrans)
      else:
         loss, preds = simple_loss(outputs, labels, critStage)

      loss.backward()
      optimizer.step()

      pbar.update(i)

      loss_acm += loss.data[0]
      acc_acm += torch.sum(preds.data == labels.data)
      del frames, times, labels, trans, loss

   pbar.finish()

   return acc_acm, loss_acm


def eval_epoch(model, val_loader, critStage, critTrans, conf):
   model.eval()
   vloss_acm = 0
   vacc_acm = 0
   for frames, times, labels, trans in val_loader:
      frames = [Variable(frame.cuda()) for frame in frames]
      labels = Variable(labels.cuda())
      trans = Variable(trans.cuda()).view(-1, 1).float()
      times = [Variable(time.cuda()).view(-1, 1).float() for time in times]
      outputs = model(frames, times)

      loss, preds = simple_loss(outputs, labels, critStage)

      vloss_acm += loss.data[0]
      vacc_acm += torch.sum(preds.data == labels.data)

      del frames, times, labels, trans, loss

   return vacc_acm, vloss_acm


def train(conf):

   if 'cut' in conf['dataset']:
      numLabels = 6
   else:
      numLabels = 15
   train_dataset, val_dataset, train_loader, val_loader = load_datasets(conf)
   if conf['conf_file'] == 'resnet50':
      model = embryoNet(numLabels, False, False, conf['in_ch'], 50)
   elif conf['conf_file'] == 'resnet18':
      model = embryoNet(numLabels, False, False, conf['in_ch'], 18)
   elif conf['conf_file'] == 'lateFusion50':
      model = lateFusion(numLabels, False, False, conf['in_ch'], 50, conf['num_conv'])
   elif conf['conf_file'] == 'lateFusion18':
      model = lateFusion(numLabels, False, False, conf['in_ch'], 18, conf['num_conv'])
   elif conf['conf_file'] == 'lateFusionFrozen':
      model = lateFusionFrozen(numLabels, False, False, conf['in_ch'], 18, conf['num_conv'])
   else:
      model = cnn('/home/nathan/torchlib/config/' + conf['conf_file'])

   if(conf['loss'] == 'xentropy'):
      critStage = nn.CrossEntropyLoss()
      critTrans = nn.BCEWithLogitsLoss()

   if('Frozen' in conf['conf_file']):
      params = [{'params': model.conv.parameters()},
                {'params': model.fc.parameters()}]
   else:
      params = model.parameters()

   if conf['optim'] == 'SGD':
      optimizer = torch.optim.SGD(params, lr=conf['lr'])
   elif conf['optim'] == 'Adam':
      optimizer = torch.optim.Adam(params, lr=conf['lr'])

   model.cuda()
   model = nn.DataParallel(model)

   if conf['optim'] == 'Adagrad':
      optimizer = torch.optim.Adagrad(params, lr=conf['lr'])
   model_dir = (
         '/data2/nathan/embryo/model/' + conf['conf_file'] + '/' +
         str(conf['lr']) + 'lr_' + str(conf['in_ch']) + 'inch_' + str(conf['optim']) + '/'
   )
   if not os.path.exists(model_dir):
      os.makedirs(model_dir)

   h_filename = model_dir + 'history'
   if os.path.exists(h_filename):
      write = 'a'
   else:
      write = 'w'
   hist = open(h_filename, write)

   if(conf['resume'] != -1):
      model.load_state_dict(torch.load(model_dir + str(conf['resume']) + '.model'))



   num_batches = len(train_dataset)/conf['batch_size']
   prev_loss = 1.0
   decay = 0
   for epoch in range(conf['resume'] + 1, conf['nb_epoch']):
      adjust_learning_rate(optimizer, decay, epoch, conf)
      acc_acm, loss_acm = train_epoch(model, train_loader, optimizer, critStage, critTrans, num_batches, conf)
      vacc_acm, vloss_acm = eval_epoch(model, val_loader, critStage, critTrans, conf)
      epoch_tacc = (1.0*acc_acm)/len(train_dataset)
      epoch_loss = (1.0*loss_acm)/len(train_dataset)
      epoch_vacc = (1.0*vacc_acm)/len(val_dataset)
      epoch_vloss = (1.0*vloss_acm)/len(val_dataset)

      print ('Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f, Train Acc: %.4f, Val Acc: %.4f'
            %(epoch, conf['nb_epoch'], epoch_loss, epoch_vloss, epoch_tacc, epoch_vacc))
      hist.write('Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f, Train Acc: %.4f, Val Acc: %.4f\n'
            %(epoch, conf['nb_epoch'], epoch_loss, epoch_vloss, epoch_tacc, epoch_vacc))
      if epoch_vloss > prev_loss:
         decay += 1
      prev_loss = epoch_vloss
      torch.save(model.state_dict(), model_dir + str(epoch) + '.model')
      model.train(True)

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
   parser.add_argument('-ha', action='store', dest='h_rate', type=int,
         help="the number of epochs to halve", default=10)
   parser.add_argument('-r', action='store', dest='resume', type=int,
         help="the epoch to resume training at", default=-1)
   parser.add_argument('-o', action='store', dest='optim', type=str,
         help='the optimizer to use', default='SGD')
   args = vars(parser.parse_args())

   train(args)
