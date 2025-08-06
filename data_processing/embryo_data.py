# dataset/embryo_dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class EmbryoDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['Class'])
        if self.transform:
            image = self.transform(image)
        return image, label
