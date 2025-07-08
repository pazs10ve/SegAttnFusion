from experiments.loaders import get_dataloaders
from src.model import get_model

import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision.transforms import v2



def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train() 
    running_loss = 0.0
    total = 0

    for idx, (inputs, labels) in enumerate(train_loader):
        print(f'{idx+1}/{len(train_loader)}')
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.float(), labels.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def eval(val_loader, model, criterion, device):
    model.eval() 
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            print(f'{idx+1}/{len(val_loader)}')
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.float(), labels.float())

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def train(train_loader, val_loader, model, criterion, optimizer, num_epochs, device, scheduler=None):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = eval(val_loader, model, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step()

    return train_losses, val_losses



"""
path = 'data'
transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
batch_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
lr = 0.001
num_epochs = 10
model = get_model(device = device)
optimizer = optim.Adam(model.parameters(), lr = lr)

train_loader, val_loader, test_loader = get_dataloaders(path, batch_size=batch_size, transform=transform)


train_losses, val_losses = train(
    train_loader, val_loader, model, criterion, optimizer, num_epochs, device
"""
