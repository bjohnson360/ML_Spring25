import argparse
import os
from detect_faces import detect_faces
from recognize_faces import get_embeddings
from utils.matcher import FaceMatcher
from PIL import ImageDraw

def main(image_path, add_name=None, threshold=0.65):
    # uses MTCNN to detect faces
    boxes, image = detect_faces(image_path)
    faces = [image.crop(box.tolist()) for box in boxes]

    if not faces:
        print("No faces detected.")
        return

    # passes cropped faces to GoogLeNetEmbedder --> 128D vector per face
    embeddings = get_embeddings(faces)

    # loads face recognition engine and any saved embeddings from disk
    matcher = FaceMatcher(threshold=threshold)
    matcher.load("data/known_faces.pkl")

    # if add_name is passed, save new identity
    if add_name:
        print(f"Saving new identity: {add_name}")
        matcher.add(add_name, embeddings[0])  # Add first face only
        matcher.save("data/known_faces.pkl")
        return

    # Match detected faces against known embeddings
    # annotates each image w/ boxes and predicted names
    draw = ImageDraw.Draw(image)
    for i, emb in enumerate(embeddings):
        name = matcher.match(emb)
        print(f"Face {i + 1}: {name}")
        draw.rectangle(boxes[i].tolist(), outline="red", width=2)
        draw.text((boxes[i][0], boxes[i][1] - 10), name, fill="red")

    image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--add_name', help='If passed, adds this name as a known identity')
    parser.add_argument('--threshold', type=float, default=0.65)
    args = parser.parse_args()

    main(args.image, add_name=args.add_name, threshold=args.threshold)
