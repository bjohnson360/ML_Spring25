import os
from detect_faces import detect_faces
from recognize_faces import get_embeddings
from utils.matcher import FaceMatcher
from PIL import ImageDraw
import matplotlib.pyplot as plt

# Recursively gather all .jpg files in FDDB
fddb_root = 'data/Dataset_FDDB/images'
fddb_imgs = []
for root, _, files in os.walk(fddb_root):
    for file in files:
        if file.endswith('.jpg'):
            fddb_imgs.append(os.path.join(root, file))

# Load matcher (should already have known identities saved)
matcher = FaceMatcher(threshold=0.7)
matcher.load("data/known_faces.pkl")

print(f"Running on {len(fddb_imgs[:5])} FDDB images...\n")

# Process the first 5 images for testing
for path in fddb_imgs[:5]:
    boxes, img = detect_faces(path, threshold=0.7)
    faces = [img.crop(box.tolist()) for box in boxes]

    embeddings = get_embeddings(faces)
    draw = ImageDraw.Draw(img)

    print(f"\n{os.path.basename(path)} - Found {len(boxes)} face(s):")

    for i, (emb, box) in enumerate(zip(embeddings, boxes)):
        name = matcher.match(emb)
        print(f"  Face {i + 1}: {name}")
        draw.rectangle(box.tolist(), outline="lime", width=2)
        draw.text((box[0], box[1] - 10), name, fill="lime")

    plt.imshow(img)
    plt.title(f"{os.path.basename(path)} — Detected: {len(boxes)}")
    plt.axis("off")
    plt.show()
