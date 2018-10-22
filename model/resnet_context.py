import torch
import torch.nn as nn

class resnet_context(nn.Module):

   def __init__(self, resnet, num_frames):

def load_model():
   print("Loading Model...")
   models.resnet.model_urls['resnet50'] = 'http://download.pytorch.org/models/resnet50-19c8e357.pth'
   resnet = models.resnet50(pretrained=True)
   # replace the last output with 7
   num_ftrs = resnet.fc.in_features
   # model.fc = nn.Linear(num_ftrs, 7)
   c1_old_weights = resnet.state_dict()["conv1.weight"].numpy()
   c1_new_weights = np.concatenate((c1_old_weights, c1_old_weights, c1_old_weights), axis=1)

   resnet.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
   resnet.state_dict()["conv1.weight"] = torch.from_numpy(c1_new_weights)

   # remove last layer to add in custom time feature
   resnet.fc = nn.Sequential()
   #for param in resnet.parameters():
   #   param.requires_grad = False
   #for param in resnet.fc.parameters():
   #   param.requires_grad = True
   model = timeres(resnet, num_ftrs)
   model.cuda()
   model = nn.DataParallel(model)
   #resnet.load_state_dict(torch.load('/home/nathan/resnet1e-0510.model'))
   print("Model loaded")
   return model

