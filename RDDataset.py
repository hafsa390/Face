from torch.utils.data import Dataset, DataLoader
import matplotlib.image as img
import os
import numpy as np
from  PIL import Image

class RDDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = Image.open(img_path).convert('RGB')
        image = np.asarray(image)
        if self.transform is not None:
            image = self.transform(np.uint8(image))

        return image, label

