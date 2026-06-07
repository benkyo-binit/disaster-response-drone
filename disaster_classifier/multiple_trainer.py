import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Set seed for reproducibility
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define training and validation logic
def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        val_loss, val_correct = 0.0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"📅 Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history

# Prepare dataset loaders
def get_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Plot comparison graph
def plot_comparisons(histories, metric):
    for name, hist in histories.items():
        plt.plot(hist[metric], label=name)
    plt.title(f'{metric.replace("_", " ").title()} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid(True)
    plt.show()

# Main training loop
if __name__ == '__main__':
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    data_dir = './dataset'  # Path to your dataset
    batch_size = 32
    epochs = 30
    lr = 1e-4

    models_to_train = {
        'ResNet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        'ResNet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'VGG16': models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        'MobileNetV2': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        'EfficientNetB0': models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    }

    train_loader, val_loader = get_loaders(data_dir, batch_size)
    num_classes = len(train_loader.dataset.dataset.classes)

    histories = {}
    for name, model in models_to_train.items():
        print(f"\n🚀 Training {name}...")
        if 'resnet' in name.lower() or 'mobilenet' in name.lower() or 'efficientnet' in name.lower():
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vgg' in name.lower():
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        trained_model, history = train_model(model, train_loader, val_loader, epochs, lr, device)
        histories[name] = history

    # Plot all comparisons
    plot_comparisons(histories, 'val_acc')
    plot_comparisons(histories, 'val_loss')
