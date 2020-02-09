# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt

def run_sceneryCNN_basic(img_name='~/Documents/Insight/greenRoute/TreeRoutes/data/sond/val/1/846527.jpg',modelFilePath='./'): # random default
    # load the pre-trained weights
    model_file = modelFilePath+'/resnet18_latest.pth.tar'

    model = models.__dict__['resnet18'](num_classes=10)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_name)
    try:
        input_img = V(centre_crop(img).unsqueeze(0))
    except:
        return None

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    
    # output the prediction
    prediction=0.
    scale=float(sum(probs)) # normalize probabilities
    for iP,pred in enumerate(idx):
        prediction+=float(probs[iP])*int(pred)
    
    prediction=prediction/scale

    return prediction
    
if __name__=='__main__':
    run_sceneryCNN_basic()

    
# # predicted vs actual scenery score
# plt.plot(actual,predicted)
# plt.xlabel('Actual Scenery Score')
# plt.ylabel('Predicted Scenery Score')
# plt.savefig('predictedVsActual.pdf')
# 
# plt.clf()
# # (predicted - actual) vs actual
# plt.plot(actual,(predicted-actual))
# plt.xlabel('Actual Scenery Score')
# plt.ylabel('(Predicted-Actual) Scenery Score')
# plt.savefig('diffVsActual.pdf')
    
