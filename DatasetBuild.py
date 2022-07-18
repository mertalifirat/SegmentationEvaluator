import torch
from torch.utils.data import Dataset
import os
import numpy as np
import mmcv
from PIL import Image
class CustomImageDataset(Dataset):
    ### Custom dataset class which is inherited from the torch.utils.data.Dataset
    def __init__(self, img_dir, ann_dir,img_suffix,ann_suffix, CLASSES,transform):
        ### Custom class constructor
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.transform=transform
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.img_infos=self.loadImgInfos(self.img_dir,self.ann_dir,self.img_suffix,self.ann_suffix)
        self.file_client_args={'backend': 'disk'}
        self.file_client=None
        self.color_type='color'
        self.imdecode_backend='cv2'
        self.numClass=len(CLASSES)
        self.CLASSES=CLASSES
        self.originalShape=None

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        ### returns the transformed data, and annotation path which refers to the transformed data
        img_path = self.img_infos[idx]['filename']
        img_path=self.img_dir+'/'+img_path
        annotation = self.img_infos[idx]['ann']['seg_map']
        annotation=self.ann_dir+'/'+annotation 
        if self.transform is not None:
            image = Image.open(img_path)
            return self.transform(image),annotation
        else:
            return self.transformFunc(img_path),annotation 
    
    def transformFunc(self,img_path):
        ### Transforming function, it takes a path of the image and loads it and returns the transformed version of it
        imgg=self.imgfrombytes(img_path) ### reads image and returns bytes
        self.originalShape=imgg.shape[:2] ### stroing the original shape of the per image
        imgg = mmcv.imrescale( imgg,(2048, 512) , return_scale=False) ### Rescale function from mmcv library
        imgg = imgg.astype(np.float32) 
        imgg=torch.Tensor(imgg)
        imgg = mmcv.imnormalize(imgg.numpy(),np.float32([123.675, 116.28 , 103.53]), np.float32([58.395 ,57.12,  57.375]), True) ### normalization with given parameters 
        imgg=self.imgToTensor(imgg) 
        return imgg   

    def loadImgInfos(self, img_dir, ann_dir, img_suffix, ann_suffix):
        ### loading infos of the images and annotations which will be used in __getitem__
        img_infos=[]
        temp=[]
        for _,_,i in os.walk(img_dir):
            temp.append(i)
        temp=temp[0]
        for img in temp:
            img_info = dict(filename=img)
            img_info['ann'] = dict(seg_map=img)
            img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        return img_infos

    def imgfrombytes(self,filename):
        ### opening the per image to bytes
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
        img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        return img

    def imgToTensor(self,img):
        ### convert the img with to tensor and transposing as wanted shape
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if isinstance(img, torch.Tensor):
            img=img.numpy()
            return torch.from_numpy(img.transpose(2, 0, 1))
        elif isinstance(img, np.ndarray):
            return torch.from_numpy(img.transpose(2, 0, 1))
        else :
            return img  

    def loadSegMap(self,ann_path):
        ### loading annotation from given path per image returns the segmentation map
        self.file_client = mmcv.FileClient(**self.file_client_args)
        filename=ann_path
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend='pillow').squeeze().astype(np.uint8)
        return gt_semantic_seg



