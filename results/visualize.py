import os
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from experiments.loaders import get_dataloaders
from torchvision.transforms import v2
from transformers import AutoTokenizer


def plot(images, labels):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    fig = plt.figure(figsize=(12, 8))
    spec = gridspec.GridSpec(2, 2, width_ratios=[2, 3]) 

    for idx, (image, label) in enumerate(zip(images, labels)):
        if idx >= 2:  
            break
        
        label_text = tokenizer.decode(label.tolist(), skip_special_tokens=True)

        ax_img = fig.add_subplot(spec[idx, 0])
        ax_img.imshow(image.permute(1, 2, 0).cpu())
        ax_img.axis('off')

        ax_text = fig.add_subplot(spec[idx, 1])
        ax_text.axis('off')
        ax_text.text(0, 1, label_text, fontsize=10, va='top', wrap=True)

    plt.tight_layout()
    plt.show()



def visualize_dataset(data_dir):
    transform = v2.Compose([
        v2.Resize((512, 512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=32, transform=transform)
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
    #plot(images, labels)

    #images, labels = next(iter(val_loader))
    #plot(images, labels)

    #images, labels = next(iter(test_loader))
    #plot(images, labels)


if __name__ == "__main__":
    data_dir = "data"
    visualize_dataset(data_dir)
