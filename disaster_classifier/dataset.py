from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from config import IMAGE_SIZE, IMAGE_FOLDER



class DisasterDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(IMAGE_FOLDER, row['path'])  # 'path' column contains relative image path
        image = Image.open(image_path).convert('RGB')
        label = row['label']
        return self.transform(image), label
