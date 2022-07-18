import helperFuncs
from DatasetBuild import CustomImageDataset
import torch 
from torch.utils.data.dataloader import DataLoader
###############
#### TODO #####
model=torch.jit.load('') ### .pt file of your model state_dict
CLASSES=[] ### specify the classes  ==>>  [str]
ImagesPath='' ### give the path of your images directory that will be used on testing ==>> str
AnnotationsPath='' ### give path of your annotations directory that will be used on testing ==>> str
imgSuffix='' ### image file suffix (e.g. .png) ==>> str
annSuffix='' ### annotation file suffix (e.g .png) ==>> str
transform=None ### (optional) custom transforms ==> transforms.Compose[] , if not given default fcuntion will be executed for transforming
###############
###############
dataset=CustomImageDataset(ImagesPath,AnnotationsPath,imgSuffix,annSuffix,CLASSES,transform)
dataload=DataLoader(dataset)
precision,recall,IoU=helperFuncs.evaluate(model,dataset,dataload)
### EDITABLE ###
### evaluate function prints and returns the metric values, it is up to user to use it as void or not ###
################
