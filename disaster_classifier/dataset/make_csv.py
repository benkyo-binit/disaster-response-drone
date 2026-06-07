import os
import csv

# Set path to your dataset root
dataset_dir = 'disaster_frames'
output_csv = 'dataset.csv'

# Prepare the list to store rows
data = [
            ['collapsed_building/collapsed_building_image0001.jpg', 'collapsed_building'],
            ['fire/fire_image0001.jpg', 'fire'],
            ['flood/flood_image0001.jpg', 'flood'],
            ['normal/normal_image0001.jpg', 'normal'],
            ['traffic_incident/traffic_incident_image0001.jpg', 'traffic incident']
]

# Walk through each subfolder (label)
for label in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, label)
    
    # Ensure it's a folder (not a file)
    if not os.path.isdir(class_dir):
        continue

    # Walk through each image in that folder
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            relative_path = os.path.join(label, filename)  # folder/image.jpg
            data.append([relative_path, label])

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'label'])  # Header
    writer.writerows(data)

print(f"✅ CSV saved to: {output_csv}")
print(f"🖼️ Total labeled images: {len(data)}")
