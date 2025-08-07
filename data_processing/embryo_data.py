# dataset/embryo_dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class EmbryoDataset(Dataset):
    def __init__(self, csv_or_df, image_dir, transform=None, is_test=False, df_is_path=True):
        if df_is_path:
            self.df = pd.read_csv(csv_or_df)
        else:
            self.df = csv_or_df.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['Class'])
        if self.transform:
            image = self.transform(image)
        return image, label
