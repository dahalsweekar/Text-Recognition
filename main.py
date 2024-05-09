import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import sys

# sys.path.append("/content/drive/MyDrive/Recognition/")

from services.demo import Demo
from services.train import Train
from services.evaluate import Eval
from utils.model import Net
from utils.dataset import Dataset
from utils.visualizer import Visualize
from utils.plot_acc_loss import Plot
import torchvision.transforms as transforms
from utils.model_summary import Summary
import warnings

warnings.filterwarnings('ignore')
sys_path = os.getcwd()


def main():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--demo', help='run demo', default=False, action='store_true')
    parser.add_argument('--train', help='start training', default=False, action='store_true')
    parser.add_argument('--eval', help='start evaluation', default=False, action='store_true')
    parser.add_argument('--visualize', help='visualize sample data', default=False, action='store_true')
    parser.add_argument('--plot', help='plot accuracy and loss', default=False, action='store_true')
    parser.add_argument('--epoch', type=int, help='number of epochs', default=50)
    parser.add_argument('--batch_size', type=int, help='set batch size', default=32)
    parser.add_argument('--weights', help='set weights path', default='model/model.pth')
    parser.add_argument('--dataset', help='set dataset directory', default='dataset/crops')
    parser.add_argument('--labels', help='set label path', default='dataset/labels.csv')
    parser.add_argument('--test_image', help='set single image path', default='dataset/crops/27.jpg')

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    Main(args['demo'], args['train'], args['eval'], args['epoch'], args['batch_size'], args['weights'],
         args['dataset'], args['labels'], args['test_image'], args['visualize'], args['plot']).run()


class Main:
    def __init__(self, demo, train, evaluate, epoch, batch_size, weights, dataset, labels, test_image, visualize, plot):
        self.demo = demo
        self.train = train
        self.evaluate = evaluate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weights = weights
        self.dataset = os.path.join(sys_path, dataset)
        self.labels = os.path.join(sys_path, labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_image = os.path.join(sys_path, test_image)
        self.visualize = visualize
        self.plot = plot

    def run(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_loader, test_loader, num_classes, encode_labels = Dataset(
            annotation_file=self.labels,
            img_dir=self.dataset,
            transform=transform,
            batch_size=self.batch_size).dataloader()
        print(f'Total {num_classes} classes found.')
        model = Net(num_classes=num_classes)
        print('Model summary:')
        Summary(model).count_parameters()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        if self.demo:
            assert os.path.isfile(self.weights) is True, 'model weights not found'
            assert os.path.isfile(self.test_image) is True, 'sample image does not exists'
            model.load_state_dict(torch.load(self.weights))
            predictions = Demo(model=model, img_path=self.test_image,
                               transform=transform).predict()
            print(f'Predicted result: {encode_labels.get(str(predictions))}')

        if self.evaluate:
            assert os.path.isfile(self.weights) is True, 'model weights not found'
            model.load_state_dict(torch.load(self.weights))
            Eval(model=model, test_loader=test_loader, criterion=criterion, device=self.device, 
                  encode_labels=encode_labels).evaluate()

        if self.train:
            train_acc_history, train_loss_history = Train(model=model, optimizer=optimizer, criterion=criterion,
                                                          train_loader=train_loader, device=self.device,
                                                          epochs=self.epoch, model_path=self.weights).train()

        if self.visualize:
            Visualize(annotation_file=self.labels, img_dir=self.dataset).vis()

        if self.plot:
            Plot(train_acc_history, train_loss_history).plot_metrics()


if __name__ == '__main__':
    main()
