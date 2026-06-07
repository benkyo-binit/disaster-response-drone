# inference.py
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from model import get_model

# Class names (must match training)
class_names = ['collapsed_building', 'fire', 'flood', 'traffic_incident']
CONFIDENCE_THRESHOLD = 0.65

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load("disaster_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def infer(frame):
    """
    Perform disaster classification on a single video frame.
    Input:
        frame (numpy.ndarray): BGR image from OpenCV
    Output:
        predicted_class (str)
        confidence (float)
    """

    # Convert OpenCV BGR → RGB → PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Preprocess
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        max_prob, predicted = torch.max(probs, 1)

    confidence = max_prob.item()
    predicted_class = class_names[predicted.item()]

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "normal"

    return predicted_class, confidence
