import torch.nn as nn
import torch.nn.functional as F
from recognize_faces import GoogLeNetEmbedder

# PyTorch NN for Face Classification 
class FaceClassifier(nn.Module):
    # num_classes: # of people in dataset
    # embedding_dim: dimensionality of feature vector
    def __init__(self, num_classes, embedding_dim=128):
        super(FaceClassifier, self).__init__()
        # extracts 128D feature vector from each face
        self.embedder = GoogLeNetEmbedder(embedding_dim=embedding_dim)
        # linear layer that mapvs vector to class logits
        self.classifier = nn.Linear(embedding_dim, num_classes)

    # Input: a batch of face crops
    # Extracts 128D embeddings
    # Feeds into classifier
    # Returns logits (unnormalized class scores) --> used in CrossEntropyLoss
    def forward(self, x):
        embeddings = self.embedder(x)
        logits = self.classifier(embeddings)
        return logits