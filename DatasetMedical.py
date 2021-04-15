import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms
from PIL import Image


#load data from a folder
class DatasetMedical(Dataset):
    def __init__(self, gray_dir, gray_files, gt_dir, pytorch=True, transform=None, gaussian_blur=False):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(gray_dir/f, gt_dir) for f in gray_files]
        self.pytorch = pytorch
        self.transform = transform
        if gaussian_blur:
            self.gaussian_blur = transforms.GaussianBlur(11, sigma=(1.5, 2.0))
        else:
            self.gaussian_blur = False
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}
        print(files)
        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        # open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        # Gaussian Blur
        if self.gaussian_blur:
            x = self.gaussian_blur(x)
        # Transformation
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def get_transform(self):
        return self.transform
