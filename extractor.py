import torch
from torch import nn
import numpy
import torchvision
from PIL import Image 
import glob
from torch.utils.data import Dataset, DataLoader  
from torchvision.transforms import transforms
import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self,batch_size=16,device='cuda:0'):
        self.imgset = None
        self.labelset = None
        self.batch_size = batch_size
        self.device = device
        self.extractor = torchvision.models.vgg16(pretrained = True).features.to(device)
        self.extractor.eval()
        
        print('------------------- Feature Extractor -------------------')
        print(f'batch_size: {self.batch_size}')
        print(f'device: {self.device}')
        print('extractor: VGG16')
        print('---------------------------------------------------------')
        
    
    def generate_stream(self,imgset,labelset):
        dataset = NNDataset(imgset,labelset)
        dataloader = DataLoader(dataset,batch_size=self.batch_size)
        
        features = None
        labels = []
        
        with torch.no_grad():
            for _,(img,label) in enumerate(dataloader):
                img = img.to(self.device).float()
                
                feat = self.extractor(img)
                feat = feat.view(feat.size(0), -1) 
                # raise ValueError(f'Here feat is of: {feat.shape}')
                
                if features == None:
                    features = feat
                else:
                    features = torch.concat((features,feat),dim=0)
                
                labels += label.cpu().tolist()
        
        features = features.cpu().numpy()
        labels = np.array(labels)
        
        del dataset
        del dataloader
        
        return features,labels
        
        


class NNDataset(Dataset):
    def __init__(self,imgset,labelset):
        super(NNDataset,self).__init__()
        self.imgset = imgset
        self.labelset = labelset   
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self,idx):        
        img = Image.open(self.imgset[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labelset[idx]
        return img, label
        

if __name__ == '__main__':
    a = NNDataset()
