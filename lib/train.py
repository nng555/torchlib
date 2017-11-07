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


def train(conf):
   widgets = [Percentage(), ' ', Bar(marker='=',left='[',right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
   train_dataset = embryoDataset(
         'train',
         transform=transforms.Compose(
            transforms.RandomHorizontalFlip(),
            tranfsorms.RandomCrop(200, 0),
            transforms.toTensor()
         ))
   val_dataset = embryoDataset('val', transform=transforms.ToTensor())

   train_loader = DataLoader(dataset=train_dataset,
         batch_size=conf['batch_size'],
         shuffle=True)
   val_loader = DataLoader(dataset=val_dataset,
         batch_size=conf['batch_size'],
         shuffle=False)
   if conf['conf_file'] == 'resnet50':
      models.resnet.model_urls['resnet50'] = 'http://download.pytorch.org/models/resnet50-19c8e357.pth'
      model = models.resnet50(pretrained=True)
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs, 15)
   else:
      model = cnn('/home/nathan/torchlib/config/' + conf['conf_file'])

   if(conf['loss'] == 'xentropy'):
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])

   model.cuda()
   model = nn.DataParallel(model)
   num_batches = len(train_dataset)/conf['batch_size']

   for epoch in range(conf['nb_epoch']):
      pbar = ProgressBar(widgets=widgets, maxval=num_batches)
      pbar.start()
      loss_acm = 0
      acc_acm = 0
      for i, (frames, times, labels) in enumerate(train_loader):
         frames = Variable(frames.cuda())
         times = Variable(times.cuda())
         labels = Variable(labels.cuda())

         optimizer.zero_grad()
         outputs = model(frames)
         _, preds = torch.max(outputs.data, 1)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         pbar.update(i)

         #if (i+1)%100 == 0:
         #   print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
         #            %(epoch+1, conf['nb_epoch'], i+1, len(train_dataset)//conf['batch_size'], loss.data[0]))

         loss_acm += loss.data[0]
         acc_acm += torch.sum(preds == labels.data)

      epoch_loss = loss_acm/len(train_dataset)
      epoch_tacc = 1.0*acc_acm/len(train_dataset)
      model.eval()
      correct = 0
      for images, times, labels in val_loader:
         images = Variable(images.cuda())
         outputs = model(images)
         _, predicted = torch.max(outputs.data, 1)
         correct += (labels.cpu().eq(predicted.cpu().long())).sum()
      epoch_vacc = 1.0*correct/len(val_dataset)
      print ('Epoch [%d/%d], Loss: %.4f, Train Acc: %.4f, Val Acc: %.4f'
               %(epoch+1, conf['nb_epoch'], epoch_loss, epoch_tacc, epoch_vacc))
      torch.save(model.state_dict(), '/data2/nathan/embryo/model/resnet' + str(conf['lr']) + str(epoch) + '.model')
      model.train(True)
      pbar.finish()


   print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

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

   train(args)
