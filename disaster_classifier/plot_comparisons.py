import pickle
import matplotlib.pyplot as plt
import os

# Define the models and their metric files
model_files = {
    "ResNet18": "results/resnet18_metrics.pkl",
    "MobileNetV2": "results/mobilenetv2_metrics.pkl",
    "VGG16": "results/vgg16_metrics.pkl"
}

# Load all metrics
metrics_dict = {}
for model_name, filepath in model_files.items():
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            metrics_dict[model_name] = pickle.load(f)
    else:
        print(f"⚠️ Warning: {filepath} not found. Skipping {model_name}.")

# Plotting function
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(10, 5))
    for model_name, metrics in metrics_dict.items():
        plt.plot(metrics[metric_name], label=model_name)
    plt.title(f"{metric_name.replace('_', ' ').title()} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{metric_name}_comparison.png")
    plt.show()

# Plot all four metrics
plot_metric("train_acc", "Accuracy")
plot_metric("val_acc", "Accuracy")
plot_metric("train_loss", "Loss")
plot_metric("val_loss", "Loss")
