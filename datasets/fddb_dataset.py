import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FDDBFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir

        # Remap labels to a continuous range [0, N-1]
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        self.annotations['label'] = self.annotations['label'].map(self.label_map)

        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx]['label'])  # ensure label is int

        if self.transform:
            image = self.transform(image)

        return image, label