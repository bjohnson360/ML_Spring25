import torch
import torch.nn.functional as F
import pickle
import os

class FaceMatcher:
    def __init__(self, known_faces=None, threshold=0.6):
        # known_faces = {"name": embedding_tensor}
        self.known_faces = known_faces if known_faces else {}
        self.threshold = threshold  # cosine similarity cutoff

    def add(self, name, embedding):
        """Add a new embedding for a person."""
        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].append(embedding)

    def match(self, embedding):
        """Compare an embedding to all known embeddings and return best match."""
        if not self.known_faces:
            return "unknown"

        best_match = "unknown"
        best_score = -1

        for name, embeddings in self.known_faces.items():
            all_embeddings = torch.stack(embeddings)
            similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings)
            max_sim = torch.max(similarities).item()

            if max_sim > best_score and max_sim > self.threshold:
                best_score = max_sim
                best_match = name

        return best_match

    def save(self, path="data/known_faces.pkl"):
        # Save as {name: [tensor1.cpu(), tensor2.cpu()...]}
        to_save = {k: [e.cpu() for e in v] for k, v in self.known_faces.items()}
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)

    def load(self, path="known_faces.pkl"):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.known_faces = pickle.load(f)
        else:
            print(f"[WARN] No known faces file found at {path}")