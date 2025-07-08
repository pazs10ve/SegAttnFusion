import torch
import torch.nn as nn
from torchvision.transforms import v2

from utils.load_config import load_config
from experiments.loaders import get_dataloaders
from inference.upscale import upscale

from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



class MedicalImageSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        """
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)   Initialize DeepLabV3+ for medical image segmentation
        
        """
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)['out']
    
    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            return torch.argmax(output, dim=1)



def combine_image_and_mask(image, mask, alpha=0.5):
    mask_color = Image.new("RGB", mask.size)
    mask_color.paste(image, mask=mask)
    blended = Image.blend(image, mask_color, alpha=alpha) 
    return blended


def combine(images, masks):
    tensors = []
    for _, (image, mask)in enumerate(zip(images, masks)):
        orig_image = Image.fromarray((image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        pred_mask = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
        combined = combine_image_and_mask(orig_image, pred_mask, alpha=0.5)
        transform = v2.Compose([
        v2.ToImage(),
        v2.ToTensor()
    ])
        combined = transform(combined)
        tensors.append(combined)
    return torch.stack(tensors)



"""train_loader, _, _ = get_dataloaders(data_dir, batch_size=batch_size, transform=transform)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device is {device}")

model = MedicalImageSegmentor().to(device)
    
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)
print("Input images shape:", images.shape)
print("Input labels shape:", labels.shape)
print(labels)"""

"""

data_dir = 'data'
batch_size = 6
    
transform = v2.Compose([
        v2.Resize((512, 512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

train_loader, _, _ = get_dataloaders(data_dir, batch_size=batch_size, transform=transform)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device is {device}")

model = MedicalImageSegmentor().to(device)
    
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)
print("Input images shape:", images.shape)
print("Input labels shape:", labels.shape)

    
masks = model.predict(images)
print("Segmentation mask shape:", masks.shape)

segs = combine(images, masks)
print(segs.shape)
print("segmentations arrived")
segs = upscale(segs)
print(segs.shape)
"""
