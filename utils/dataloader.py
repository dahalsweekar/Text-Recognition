from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict['labels'])

    def __getitem__(self, idx):
        img = self.data_dict['images'][idx]
        label = self.data_dict['labels'][idx]
        img = Image.merge('RGB', [img, img, img])
        if self.transform:
            img = self.transform(img)

        return img, label
