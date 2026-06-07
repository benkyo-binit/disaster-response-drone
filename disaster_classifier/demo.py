# display_demo.py
import cv2
import numpy as np
import time
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


# dummy predict function — replace with your model inference
def predict(frame):
    # returns list of (label, confidence) sorted desc
    # example: [("Flood",0.84), ("Fire",0.10), ("No Disaster",0.06)]
    return [(predicted_class, confidence)]

cap = cv2.VideoCapture(0)  # or your stream URL
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret: break

    preds = predict(frame)
    top_label, top_conf = preds[0]

    # semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 80), (0,0,0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # big title
    cv2.putText(frame, f"Prediction: {top_label}", (20,35), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {top_conf*100:.1f}%", (20,65), font, 0.8, (200,200,200), 2, cv2.LINE_AA)

    # draw confidence bar
    bar_x = 400
    bar_y = 20
    bar_w = 300
    bar_h = 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w*top_conf), bar_y+bar_h), (0,200,0), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 2)

    # top-3 list
    y0 = 110
    for i, (lab,conf) in enumerate(preds[:3]):
        cv2.putText(frame, f"{i+1}. {lab} ({conf*100:.1f}%)", (20, y0 + i*30), font, 0.7, (240,240,240), 2)

    # timestamp (and fake GPS — replace if you have real)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts + "  |  GPS: 27.7N, 85.3E", (frame.shape[1]-420, frame.shape[0]-20), font, 0.6, (220,220,220), 1)

    cv2.imshow('Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
