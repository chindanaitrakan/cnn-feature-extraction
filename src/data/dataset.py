import numpy as np
import os

from torchvision import transforms, datasets
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class GrayscaleTransform:
    def __init__(self):
        pass

    def __call__(self, img):

        img = transforms.ToTensor()(img)
        img = torch.mean(img, dim=0, keepdim=True) 

        return img
    
class CIFAR10Data(Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()

        if train: self.dir = "train"
        else: self.dir = "test"
        self.data_path = './data/raw/'+self.dir

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.data = datasets.CIFAR10(root='./data/raw/'+self.dir, train=train, download=True,)
        self.img_trans = transforms.Compose(
            [
                GrayscaleTransform(),
            ]
        )
        # get eval data
        if not train:
            self.eval_image, self.eval_label = self.data[-1]
            self.eval_image = self.img_trans(self.eval_image)
            # Exclude the eval image
            self.data.data = self.data.data[:-1]  
            self.data.targets = self.data.targets[:-1]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx] 
        image = self.img_trans(image)
    
        return image, label
    
    def get_eval(self):
        return self.eval_image, self.eval_label
    

    
        
        
    
