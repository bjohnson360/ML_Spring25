import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Face crop transform
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
        base.fc = nn.Identity()
        self.backbone = base
        self.embed = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize

# Load model
model = GoogLeNetEmbedder().to(device).eval()

def get_embeddings(face_crops):
    """ Generate embeddings for a list of face crops (PIL images). """
    tensors = [crop_transform(face).unsqueeze(0).to(device) for face in face_crops]
    batch = torch.cat(tensors, dim=0)

    with torch.no_grad():
        embeddings = model(batch)

    return embeddings.cpu()
