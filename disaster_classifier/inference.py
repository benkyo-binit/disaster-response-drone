import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Class names (must match training)
class_names = ['collapsed_building', 'fire', 'flood', 'traffic_incident']
CONFIDENCE_THRESHOLD = 0.5 

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load("disaster_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Image transform (match your training transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB normalization
])

# Load image
image_path = "test_image.jpg"  # or change to any test image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Inference with confidence
with torch.no_grad():
    output = model(image)
    probs = torch.nn.functional.softmax(output, dim=1)
    max_prob, predicted = torch.max(probs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = max_prob.item()

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "normal"

print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

