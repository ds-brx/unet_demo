import os
import torch
import numpy as np
import pandas as pd
from dataframe import cityscapes_df
from dataset import cityscapes_dataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from unet_model import UNet
from tqdm import tqdm

train_img_dir = "/Users/breenda/Desktop/unet/imgs/train"
train_mask_dir = "/Users/breenda/Desktop/unet/masks/train"
valid_img_dir = "/Users/breenda/Desktop/unet/imgs/val"
valid_mask_dir = "/Users/breenda/Desktop/unet/masks/val"

train_df = pd.DataFrame(columns=['img_file_name','mask_file_name','subdir'])
valid_df = pd.DataFrame(columns=['img_file_name','mask_file_name','subdir'])

for t, v in zip(os.listdir(train_img_dir), os.listdir(valid_img_dir)):
    train_df = cityscapes_df(train_df, train_img_dir, train_mask_dir, t)
    valid_df = cityscapes_df(valid_df, valid_img_dir, valid_mask_dir, v)

train_df.to_csv("train_df.csv")
valid_df.to_csv("valid_df.csv")

train_dataset = cityscapes_dataset("train_df.csv", train_img_dir, train_mask_dir)
valid_dataset = cityscapes_dataset("valid_df.csv", valid_img_dir, valid_mask_dir)

train_dataloader = DataLoader(train_dataset, batch_size =1, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size =1, shuffle=False, num_workers=0)

learning_rate =  1e-5
model = UNet(n_channels=3, n_classes=35, bilinear=False)
# print(model)
n_train = len(train_df)
batch_size = 32
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
criterion = nn.CrossEntropyLoss()


def evaluate(model, device, dataloader):
    print("Validating")
    model.eval()
    num_val_batches = len(dataloader)
    for batch in tqdm(dataloader):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        with torch.no_grad():
            mask_pred = model(image)
            mask_true = mask_true[:,0,:,:]
            loss = criterion(mask_pred, mask_true)
            print("Valid Loss: {}".format(loss))

    model.train()
    return loss/num_val_batches

def train(device,epochs):

    for epoch in range(epochs):
        model.to(device=device)
        model.train()
        print("Training")
        for batch in tqdm(train_dataloader):
            images = batch['image']
            true_masks = batch['mask']
            # print(images.shape)
            # print(true_masks.shape)

            assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            masks_pred = model(images)
            # print(masks_pred.shape)
            # print(true_masks.shape)        

            # masks_pred = torch.unsqueeze(masks_pred, 0)
            true_masks = true_masks[:,0,:,:]
            print(true_masks.shape)

            # print(true_masks.shape) 
            loss = criterion(masks_pred,device, true_masks)
            print("Train Loss: {}".format(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch%10 == 0:
                val_score = evaluate(model, valid_dataloader)
                print("Validation Loss: {}".format(val_score))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(epochs=100, device=device)
        
