import helperF
from DatasetBuild import CustomImageDataset
import torch 
from torch.utils.data.dataloader import DataLoader
###############
#### TODO #####
model=torch.load('') ### .pt file of your model state_dict
CLASSES=[] ### specify the classes
ImagesPath='data/data/IJRR17/images/dir_002' ### give the path of your images directory that will be used on testing
AnnotationsPath='data/data/IJRR17/annotations/dir_002' ### give path of your annotations directory that will be used on testing
imgSuffix='' ### image file suffix (e.g. .png)
annSuffix='' ### annotation file suffix (e.g .png)
###############
###############
dataset=CustomImageDataset(ImagesPath,AnnotationsPath,imgSuffix,annSuffix,CLASSES)
dataload=DataLoader(dataset)
precision,recall,IoU=helperFuncs.evaluate(model,dataset,dataload)
