import pandas as pd
import makeCsvWithLocalPath # to make csv with local path and download any missing jpgs
import train_placesCNN # this is where intial training was done
import scenicScoreDataset # to accommodate our data structure

from sklearn.model_selection import train_test_split
import os,sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt

debug=False
evaluate=False
best_prec1=0

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def runRetrain():
  # 1. load the data
  if not os.path.exists('data/newImagesWithJpg.tsv'):
    print ("ERROR! You need to make the correct input dataset (scenic or not data with local paths). Run `python makeCsvWithLocalPath.py` (it takes quite some time) and try again." )
    sys.exit(1)

  #scenicDF = pd.read_csv('data/imagesWithJpg.csv')
  scenicDF = pd.read_csv('data/newImagesWithJpg.tsv',sep='\t')
  scenicDF = scenicDF[['Images','Average']]
  # cleaning is done in creation of this csv

  # hack to accommodate the slow download of files. remove later!
  images,averages = [],[]
  for idx,row in scenicDF.iterrows():
    if os.path.exists(row.Images):
      images.append(row.Images)
      averages.append(row.Average)

  scenicDF = pd.DataFrame(list(zip(images,averages)),columns=['Images','Averages'])
  
  #  - split into train and test groups
  image_train, image_val, score_train, score_val = train_test_split(
    scenicDF.Images,scenicDF.Average, test_size=0.2)
  
  
  # 2. load data into pytorch-friendly format
  #  - format everything to match original training
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  myTrainTransforms = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
      ])
  myValidationTransforms = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
      ])
  
  training_set = scenicScoreDataset.ScenicScoreDataset(image_train,score_train,transform=myTrainTransforms,resizeImage=True)
  train_loader = torch.utils.data.DataLoader(training_set,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=1,
                                             pin_memory=True)
  
  validation_set = scenicScoreDataset.ScenicScoreDataset(image_val,score_val,transform=myValidationTransforms,resizeImage=True)
  val_loader = torch.utils.data.DataLoader(validation_set,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=1,
                                           pin_memory=True)
  
  

  # 2. Create the base model from the pre-trained model
  #cudnn.benchmark = True
  
  arch = 'resnet18' # using the resnet architecture
  baseModelFile = '%s_places365.pth.tar' % arch # using the places365 model
  if not os.access(baseModelFile, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + baseModelFile
    os.system('wget ' + weight_url)
     
  #device = torch.device('cpu') # on my laptop. update when on aws!
  baseModel = models.__dict__[arch](num_classes=365)
  baseModel = torch.nn.DataParallel(baseModel)
  #checkpoint = torch.load(baseModelFile,  map_location=device) # same here, update on aws!
  checkpoint = torch.load(baseModelFile)
  
  baseModel.load_state_dict(checkpoint['state_dict'])
  
  # 3. Set up retraining
  # freeze all layers
  for param in baseModel.module.parameters():
    param.requires_grad = False
  
  if debug:
    print (baseModel)
  # update the fully connected layer
  nInF,nOutF = baseModel.module.fc.in_features,1 # input size from original; 1 output
  baseModel.module.fc = torch.nn.Linear(nInF,nOutF)
  
  for param in baseModel.module.fc.parameters():
    param.requires_grad = True
  
  
  #criterion = torch.nn.MSELoss() # regression mean squared loss
  criterion = torch.nn.MSELoss().cuda() # regression mean squared loss
  # copied from places training
  lr = 0.0001 # learning rate
  momentum = 0.9
  weight_decay = 1e-4
  optimizer = torch.optim.SGD(baseModel.parameters(), lr)#,
  #                            momentum=momentum,
  #                            weight_decay=weight_decay)
  #cudnn.benchmark = True # what does this do?
  
  
  # https://discuss.pytorch.org/t/pytorch-cnn-for-regression/33436/3: 
  # I suspect that the only thing I need to do different in a regression problem in Pytorch
  # is change the cost function to MSE.
  # Probably you would also change the last layer to give the desired number of outputs
  # as well as remove some non-linearity on the last layer such as F.log_softmax (if used before).
  # The activation functions between the layers should still be used.
  # I would try to use pretty much the same architecture besides
  # the small changes necessary for regression.
  
  if debug:
    print (baseModel)
  
  # 4. Run validation or training
  
  if evaluate:
    prec1, lossV,actual,predicted = train_placesCNN.validate(val_loader, baseModel, criterion)
  
  else:
    lossesT,lossesV = [],[] # for plotting
    nEpochs=90
    for epoch in range(0, nEpochs):
      train_placesCNN.adjust_learning_rate(optimizer, epoch)
  
      # train for one epoch
      thisLoss,actual,predicted = train_placesCNN.train(train_loader, baseModel, criterion, optimizer, epoch)
      lossesT.append(np.sqrt(float(thisLoss.val)))
  
      # evaluate on validation set
      prec1,losses,actual,predicted = train_placesCNN.validate(val_loader, baseModel, criterion)
      lossesV.append(np.sqrt(float(losses.val)))
      
      # remember best prec@1 and save checkpoint
      is_best = prec1 > best_prec1
      best_prec1 = max(prec1, best_prec1)
      train_placesCNN.save_checkpoint({
          'epoch': epoch + 1,
          'arch': arch,
          'state_dict': baseModel.state_dict(),
          'best_prec1': best_prec1,
          'losses' : losses,
          }, is_best, 'sceneryScore_latest.pth.tar')
          
  # 5. Plot stuff
          
      # loss vs epoch (only for training, not for validation-only processing)
    epochs = [it for it in range(0,nEpochs)]
    plt.plot(epochs,lossesT, 'b', epochs,lossesV, 'g')
  plt.xlabel('Epoch')
  plt.ylabel('Root Mean Squared Error')
  plt.savefig('errorVsEpoch.pdf')
    
  # predicted vs actual scenery score
    
  # (predicted - actual) vs actual
    
    
if __name__=='__main__':
  runRetrain()
