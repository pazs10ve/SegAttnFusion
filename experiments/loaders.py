import os
import json
from PIL import Image
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM


class IUXrayDataset(Dataset):
    def __init__(self, data_dir : str, split : str, splits : dict, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.splits = splits
        self.annotations = self.load_annotations()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")



    def get_image_paths(self, split):
        image_paths = []
        for _, row in self.annotations.iterrows():
            if row['img_path1'] is not None:
                image_paths.append((os.path.join(self.data_dir, 'images', row['img_path1']), row['category']))
            if row['img_path2'] is not None:
                image_paths.append((os.path.join(self.data_dir, 'images', row['img_path2']), row['category']))

        split_idx = int(len(image_paths) * self.splits[split])
        if split == 'train':
            return image_paths[:split_idx]
        elif split == 'val':
            return image_paths[split_idx:split_idx + int(len(image_paths) * self.splits['val'])]
        else:
            return image_paths[split_idx + int(len(image_paths) * self.splits['val']):]

    def load_annotations(self):
        json_file_path = os.path.join(self.data_dir, 'annotation.json')
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        rows = []
        for category, items in data.items():
            for item in items:
                rows.append({
                    "id": item["id"],
                    "img_path1": item["image_path"][0] if len(item["image_path"]) > 0 else None,
                    "img_path2": item["image_path"][1] if len(item["image_path"]) > 1 else None,
                    "report": item["report"],
                    "category": category
                })

        df = pd.DataFrame(rows)
        df = df[df['category'] == self.split]
        return df
    

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, idx):

        row = self.annotations.loc[idx]
    
        img_path1 = os.path.join(self.data_dir, 'images', row['img_path1'])
        img_path2 = os.path.join(self.data_dir, 'images', row['img_path2'])

        img1 = Image.open(img_path1).convert('RGB') 
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1) 
            img2 = self.transform(img2) 

    
        #combined_image = torch.cat((img1, img2), dim=-1)

        report_text = row['report']
        report_tokens = self.tokenizer(
            report_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        #print(img_path2)
        return img2, report_tokens['input_ids'].squeeze()


def get_dataloaders(data_dir : str, splits : dict = None, batch_size : int = 16, transform=None):
    splits =  {'train': 0.5, 'val': 0.2, 'test': 0.3}
    train_dataset = IUXrayDataset(data_dir, 'train', splits=splits, transform=transform)
    val_dataset = IUXrayDataset(data_dir, 'val', splits=splits, transform=transform)
    test_dataset = IUXrayDataset(data_dir, 'test', splits=splits, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




