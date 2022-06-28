import helperF
from DatasetBuild import CustomImageDataset
import torch 
from torch.utils.data.dataloader import DataLoader
model=torch.load('tmp.pt')
CLASSES=['soil','crop','weed']
dataset=CustomImageDataset('data/data/IJRR17/images/dir_002','data/data/IJRR17/annotations/dir_002',True,'png','png',CLASSES)
dataload=DataLoader(dataset)
precision,recall,IoU=helperFuncs.evaluate(model,dataset,dataload)
