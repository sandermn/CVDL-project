import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
from torchvision.transforms import ToPILImage
from medimage import image as medim

#load data from CAMUS_resized folder
class DatasetCAMUS_r(Dataset):
    def __init__(self, gray_dir, gt_dir, pytorch=True, transform=None):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        self.transform = transform
        
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}
        

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
            
    
        # normalize (raw_us.dtype = uint8)
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
        
        # transformation
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


# load data from CAMUS folder
class DatasetCAMUS(Dataset):
    """
    - Should only read mhd files from the dataset using medimage package
    - Medimage reads the files so that they can be used
    - Only use 4CH ED and ES, NOT sequence which is the full sequence of heart contraction
    """
    def __init__(self, base_path, pytorch=True, transform=None):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        #self.files = [self.combine_files(patient_dir) for patient_dir in base_path.iterdir() if not patient_dir.is_dir() or int(str(patient_dir)[-3:])>450]
        self.files = [self.combine_files(patient_dir) for patient_dir in base_path.iterdir() if int(str(patient_dir)[-3:])<=450]
        
        self.pytorch = pytorch
        self.transform = transform
        
    def combine_files(self, patient_dir: Path):
        """
        Gray and gt points to the metaheader file for each photo. 
        This file contains information about the file that can be useful
        """
        files = {'gray': medim(patient_dir/f"{str(patient_dir)[-11:]}_4CH_ED.mhd"), 
                 'gt': medim(patient_dir/f"{str(patient_dir)[-11:]}_4CH_ED_gt.mhd")}
        
        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        im = self.files[idx]['gray']
        arr = np.array(im.imdata)
        pil_im = ToPILImage()(arr)
        pil_im = self.resize_image(pil_im)
        
        raw_us = np.stack([np.array(pil_im),], axis=2)
        
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        im = self.files[idx]['gt']
        
        # create array of mask and resize
        arr = np.array(im.imdata)
        pil_im = ToPILImage()(arr)
        pil_im = self.resize_image(pil_im)
        raw_mask = np.array(pil_im) # a numpy array with unique values [0,1,2,3]
        
        # raw_mask is (384,384) array, create array with 3 channels (384,384, 3)
        
        """
        mask = np.zeros(np.concatenate((3, np.shape(raw_mask)),axis=None))
        mask[0,:,:] = mask[0,:,:] + np.where(raw_mask==1, 1, 0)
        mask[1,:,:] = mask[1,:,:] + np.where(raw_mask==2, 1, 0)
        mask[2,:,:] = mask[2,:,:] + np.where(raw_mask==3, 1, 0)
        """
        
        return raw_mask
    
    
    def resize_image(self, image):
        return image.resize((384, 384))
        
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        # transformation
        if self.transform:
            x = self.transform(x)
            #y = self.transform(y)
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def get_transform(self):
        return self.transform