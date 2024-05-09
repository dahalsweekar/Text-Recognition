import torchvision.transforms as transforms
from PIL import Image
import os
from utils.dataloader import CustomDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm


class Dataset:
    def __init__(self, annotation_file, img_dir='./dataset/crops', transform=None, batch_size=32):
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.images = []
        self.labels = []
        self.data_dict = dict()
        self.encode_labels = dict()
        self.transform = transform
        self.batch_size = batch_size

    def load_labels(self):
        with open(self.annotation_file, 'r', encoding='utf-16') as f:
            lines = f.read()
            lines = lines.split('\n')
            for row in tqdm(lines, desc='Preparing Dataset: '):
                items = row.split(' ')
                items = items[0:2]
                if len(items) == 2:
                    img_ = Image.open(os.path.join(self.img_dir, items[0]))
                    self.images.append(img_)
                    self.encode_labels[items[0][0:-4]] = items[1]
                    self.labels.append(int(items[0][0:-4]))
        self.data_dict = {'images': self.images, 'labels': self.labels}
        return self.data_dict, self.encode_labels

    def dataloader(self):
        _, _ = self.load_labels()

        dataset = CustomDataset(self.data_dict, transform=self.transform)

        train_ratio = 0.8

        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, len(dataset), self.encode_labels
