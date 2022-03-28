import pandas as pd
import os
from sqlalchemy import true

'''
Dataframe keys : img_file_name, mask_file_name, subdir
'''
def cityscapes_df(cityscapes_df, img_dir, mask_dir, subdir):
    img_file_list = [os.path.join(img_dir, subdir,f) for f in os.listdir(os.path.join(img_dir, subdir))]
    subdir_list = [subdir for _ in range(len(img_file_list))]
    mask_file_list = [os.path.join(mask_dir, subdir,f.split('/')[-1].split('leftImg8bit')[0]+"gtFine_labelIds.png") for f in img_file_list]
    dummy_df = pd.DataFrame(columns=['img_file_name','mask_file_name','subdir'])
    dummy_df['img_file_name'] = img_file_list
    dummy_df['subdir'] = subdir_list
    dummy_df['mask_file_name'] = mask_file_list
    cityscapes_df = pd.concat([cityscapes_df, dummy_df], ignore_index=true)
    return cityscapes_df



if __name__ == "__main__":
    df = pd.DataFrame(columns=['img_file_name','mask_file_name','subdir'])
    
    img_dir = "/Users/breenda/Desktop/unet/imgs/train"
    mask_dir = "/Users/breenda/Desktop/unet/masks/train"
    for f in os.listdir(img_dir):
        df = cityscapes_df(df, img_dir, mask_dir, f)
    df.to_csv("cityscapes_df.csv")
    



