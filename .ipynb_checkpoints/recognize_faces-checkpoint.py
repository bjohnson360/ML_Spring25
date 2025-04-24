import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image Preprocessing
crop_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# GoogLeNet-based embedding model
class GoogLeNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        base = models.googlenet(weights='DEFAULT', aux_logits=True)
        base.fc = nn.Identity() # get raw feature vecotrs
        self.backbone = base
        self.embed = nn.Linear(1024, embedding_dim)  # reduce feature size

    # Feeds image through GoogLeNet to get features
    # --> passes through Linear layer to get 128D vectors
    # --> normalizes to unit length (key for cosine similarity)
    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple):
            x = out[0]  # main output
        else:
            x = out
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)

# Load model
model = GoogLeNetEmbedder().to(device).eval()

# Preprocesses face crops --> converts each to 4D tensor and batches
# Outputs a tensor of shape [N, 128] where each row is a normalized embedding of a face crop
def get_embeddings(face_crops):
    """ Generate embeddings for a list of face crops (PIL images). """
    tensors = [crop_transform(face).unsqueeze(0).to(device) for face in face_crops]
    batch = torch.cat(tensors, dim=0)

    with torch.no_grad():
        embeddings = model(batch)

    return embeddings.cpu()
