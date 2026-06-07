# event_logger.py
import csv
import os
from datetime import datetime

CSV_FILE = "event_log.csv"

# Create file + header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event_type", "confidence", "source"])

def log_event(event_type, confidence, source="AI"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, f"{confidence:.2f}", source])
