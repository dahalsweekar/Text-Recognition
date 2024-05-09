import os
import matplotlib.pyplot as plt
import random
from utils.dataset import Dataset

sys_path = os.getcwd()


class Visualize:
    def __init__(self, annotation_file, img_dir='./dataset/crops'):
        self.data, self.encodes = Dataset(annotation_file, os.path.join(sys_path, img_dir)).load_labels()
        self.images = []
        self.labels = []

    def vis(self):
        start = random.randint(0, 100)
        for img in self.data['images'][start:start + 10]:
            self.images.append(img)
        for label in self.data['labels'][start:start + 10]:
            self.labels.append(label)

        fig = plt.figure(figsize=(10, 10))
        columns = 3
        rows = 3
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            print(f'keys: {self.labels[i-1]} | values: {self.encodes.get(str(self.labels[i - 1]))}')
            plt.title(f'keys: {self.labels[i-1]}')
            plt.imshow(self.images[i - 1])
        save_path = os.path.join(sys_path, 'plots/sample_dataset.png')
        plt.savefig(save_path)
        print(f'Plots save to {save_path}')
        # plt.show()
