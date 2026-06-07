#always open VSCode from root directory!!!

import torch
from torch.utils.data import DataLoader
from dataset import DisasterDataset
from data_loader import split_dataset
from model import get_model
from train import train_model
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import matplotlib.pyplot as plt

def main():
    train_df, val_df, test_df = split_dataset()

    label_to_index = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_to_index)

    train_dataset = DisasterDataset(train_df)
    val_dataset = DisasterDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    model_names = ['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']
    results = {}

    for model_name in model_names:
        print(f"\n🚀 Training model: {model_name}")
        model = get_model(model_name, num_classes=len(label_to_index))
        trained_model, history = train_model(
            model, train_loader, val_loader,
            NUM_EPOCHS, LEARNING_RATE, device,
            return_history=True  # NEW: return accuracy/loss history
        )
        torch.save(trained_model.state_dict(), f"{model_name}_classifier.pth")
        results[model_name] = history
        print(f"✅ Saved: {model_name}_classifier.pth")

    # Plot results
    plot_results(results)

def plot_results(results):
    for model_name, history in results.items():
        epochs = list(range(1, len(history['train_acc']) + 1))
        plt.plot(epochs, history['train_acc'], label=f"{model_name} (train)")
        plt.plot(epochs, history['val_acc'], label=f"{model_name} (val)")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_comparison_accuracy.png")
    plt.show()

if __name__ == "__main__":
    main()



'''
model.load_state_dict(torch.load("disaster_classifier.pth"))
model.eval()
'''