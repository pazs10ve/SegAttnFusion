import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.transforms import v2


from utils.load_config import load_config
from experiments.loaders import get_dataloaders


def visualize_feature_maps(features):
    features_to_plot = features[0, :, :, :]
    fig, axs = plt.subplots(128, 4, figsize=(15, 15))
    for i in range(128):
        for j in range(4):
            idx = i * 4 + j
            axs[i, j].imshow(features_to_plot[idx].detach().cpu().numpy(), cmap='viridis')
            axs[i, j].axis('off')
    
    #plt.tight_layout()
    plt.show()

class DenseNet121(nn.Module):
    def __init__(self, config : str = 'config.yaml'):
        super().__init__()
        #self.config = load_config(config)     
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = densenet.features[:-2]
        
    def forward(self, x):
        features = self.features(x)
        return features
    








"""data_dir = 'data'
batch_size = 6
transform = v2.Compose([
        v2.Resize((512, 512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


train_loader, _, _ = get_dataloaders(data_dir, batch_size=6, transform=transform)
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

features = model(images)
print(features.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121(config='config.yaml').to(device)
x = torch.randn(3, 3, 1024, 1024).to(device)
features = model(x)


print(f"Feature map shape: {features.shape}")
#visualize_feature_maps(features)
#features, adaptation = model(x)

"""