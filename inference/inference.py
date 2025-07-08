import torch
import torch.nn as nn
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from matplotlib import gridspec
from PIL import Image
import os
from typing import Union, List

from src.model import get_model
from experiments.loaders import get_dataloaders



def plot(images, captions):
    fig = plt.figure(figsize=(12, 8))
    spec = gridspec.GridSpec(2, 2, width_ratios=[2, 3]) 

    for idx, (image, caption) in enumerate(zip(images, captions)):
        if idx >= 2: 
            break
        
        ax_img = fig.add_subplot(spec[idx, 0])
        ax_img.imshow(image.permute(1, 2, 0).cpu() if isinstance(image, torch.Tensor) else image)
        ax_img.axis('off')

        ax_text = fig.add_subplot(spec[idx, 1])
        ax_text.axis('off')
        ax_text.text(0, 1, caption, fontsize=10, va='top', wrap=True)

    plt.tight_layout()
    plt.show()



def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = get_model(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model



def process_image(image_path: str, transform) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def run_inference_on_custom_input(
    model: nn.Module,
    image_paths: Union[str, List[str]],
    transform,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple:

    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    processed_images = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        processed_images.append(process_image(img_path, transform))
    
    image_batch = torch.stack(processed_images).to(device)    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_batch)  # Shape: [batch_size, num_classes]
        
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        
        k = 5 
        top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
        
        captions = []
        for batch_idx in range(len(image_batch)):
            tokens = top_k_indices[batch_idx]
            probs = top_k_probs[batch_idx]
            
            caption_parts = []
            for token, prob in zip(tokens, probs):
                word = tokenizer.decode(token.tolist(), skip_special_tokens=True)
                if word.strip(): 
                    caption_parts.append(f"{word} ({prob:.2f})")
            caption = " | ".join(caption_parts)
            captions.append(caption)
    
    return processed_images, captions


def run_inference(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=5):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    all_captions = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(all_captions) >= num_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            
            k = 5  
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
            
            for img_idx in range(len(images)):
                if len(all_captions) >= num_samples:
                    break
                
                tokens = top_k_indices[img_idx]
                probs = top_k_probs[img_idx]
                
                caption_parts = []
                for token, prob in zip(tokens, probs):
                    word = tokenizer.decode(token.tolist(), skip_special_tokens=True)
                    if word.strip():  
                        caption_parts.append(f"{word} ({prob:.2f})")
                caption = " | ".join(caption_parts)
                
                all_captions.append(caption)
                all_labels.append(labels[img_idx])
                all_images.append(images[img_idx])
    
    return all_images, all_labels, all_captions



def main():
    model_path = r'logs\final\1st run\models\final_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    model = load_model(model_path, device)
    
    """data_path = r'data'
    batch_size = 6
    print("Running inference on test set...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_path, 
        batch_size=batch_size, 
        transform=transform
    )
    test_images, test_labels, test_captions = run_inference(model, test_loader, device)
    
    # Plot test results
    print("\nTest Set Results:")
    print("-" * 50)
    for idx, (image, label, caption) in enumerate(zip(test_images, test_labels, test_captions)):
        print(f"Image {idx + 1}")
        print(f"Ground Truth: {label}")
        print(f"Generated Caption: {caption}")
        print("-" * 50)
    
    plot(test_images[:2], test_captions[:2])"""
    
    print("\nRunning inference on custom images...")
    custom_image_paths = [
        r'data\images\CXR5_IM-2117\1.png',
    ]
    
    try:
        custom_images, custom_captions = run_inference_on_custom_input(
            model=model,
            image_paths=custom_image_paths,
            transform=transform,
            device=device
        )
        
        print("\nCustom Image Captions:")
        print("-" * 50)
        for path, caption in zip(custom_image_paths, custom_captions):
            print(f"Image: {path}")
            print(f"Generated Caption: {caption}")
            print("-" * 50)
        
        plot(custom_images, custom_captions)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide valid image paths.")

