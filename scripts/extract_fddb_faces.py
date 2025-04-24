import os
import csv
from detect_faces import detect_faces
from PIL import Image
from pathlib import Path

# Where the FDDB images live
fddb_root = 'data/Dataset_FDDB/images'
output_dir = 'data/fddb_crops'
os.makedirs(output_dir, exist_ok=True)

# Collect all images recursively
fddb_imgs = []
for root, _, files in os.walk(fddb_root):
    for file in files:
        if file.endswith('.jpg'):
            fddb_imgs.append(os.path.join(root, file))

print(f"[INFO] Found {len(fddb_imgs)} FDDB images.")

# Labels file
labels_path = os.path.join(output_dir, 'fddb_labels.csv')
with open(labels_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])

    for img_path in fddb_imgs:
        img_id = Path(img_path).stem.replace('img_', '')  # use image number as label
        boxes, image = detect_faces(img_path, threshold=0.7)

        if boxes is None or len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            box = [int(float(coord)) for coord in box]
            cropped = image.crop(box)

            crop_filename = f"{img_id}_face{i}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cropped.save(crop_path)

            writer.writerow([crop_filename, img_id])
