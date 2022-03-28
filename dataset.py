import os
from matplotlib.animation import PillowWriter
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import PIL
from PIL import Image

imgs_train_folder = "/Users/breenda/Desktop/unet/imgs/train"
masks_train_folder = "/Users/breenda/Desktop/unet/masks/train"

class cityscapes_dataset(Dataset):

    def __init__(self, cityscapes_df, img_dir, mask_dir, transform=None):

        self.df = pd.read_csv(cityscapes_df, index_col=0)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    @classmethod
    def transform_mask(self,mask):
        new_mask = mask[:,:,0]
        class_Ids = np.array(range(-1, 34),dtype=int)
        channels = np.array([np.full((mask.shape[0], mask.shape[1]), id) for id in class_Ids])
        channels = np.array([(x==new_mask).astype(int) for x in channels])
        return channels


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.df.iloc[idx, 0]
        image = cv2.imread(img_file)
        image = image.transpose((2,0,1))
        mask_file = self.df.iloc[idx, 1]
        mask = cv2.imread(mask_file)
        mask = self.transform_mask(mask)
        sample = {'image':  torch.as_tensor(image.copy()).float().contiguous(), 'mask': torch.as_tensor(mask.copy()).long().contiguous()}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ =="__main__":
    cityscapes_df = "/Users/breenda/Desktop/unet/cityscapes_df.csv"
    train_img_dir = "/Users/breenda/Desktop/unet/imgs/train"
    train_mask_dir = "/Users/breenda/Desktop/unet/masks/train"
    train_dataset = cityscapes_dataset(cityscapes_df, train_img_dir, train_mask_dir)
    print(train_dataset[0])