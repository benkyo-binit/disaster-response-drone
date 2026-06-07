import torch
from torch.utils.data import DataLoader
from dataset import DisasterDataset
from data_loader import split_dataset
from model import get_model
from train import train_model
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

def plot_metrics(metrics, model_name):
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_acc'], label='Train Acc')
    plt.plot(epochs, metrics['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_metrics.png")
    plt.close()

def evaluate_model(model, val_loader, class_names, device, model_name):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    # Save report
    os.makedirs("results", exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(f"results/{model_name}_classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    y_scores_np = np.array(y_scores)
    
    if len(class_names) > 2:
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        for i in range(len(class_names)):
            plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve (Multiclass)')
        plt.legend(loc='lower right')
        plt.savefig(f"plots/{model_name}_roc_curve.png")
        plt.close()
    else:
        print("⚠️ Not a multiclass problem. Skipping ROC curve.")

    return report

def main():
    # Load and split dataset
    train_df, val_df, _ = split_dataset()
    label_to_index = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    train_df['label'] = train_df['label'].map(label_to_index)
    val_df['label'] = val_df['label'].map(label_to_index)
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Dataloaders
    train_dataset = DisasterDataset(train_df)
    val_dataset = DisasterDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model_name = "resnet18"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    model = get_model(model_name=model_name, num_classes=len(label_to_index))
    model.to(device)

    # Training
    trained_model, metrics = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device)
    torch.save(trained_model.state_dict(), f"{model_name}_classifier.pth")
    print(f"✅ {model_name} model saved as {model_name}_classifier.pth")

    # Plot and save
    plot_metrics(metrics, model_name)
    with open(f"results/{model_name}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # Evaluation
    evaluate_model(trained_model, val_loader, list(index_to_label.values()), device, model_name)

if __name__ == "__main__":
    main()
