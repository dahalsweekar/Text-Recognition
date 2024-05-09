import torch
import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=1):
        super(Net, self).__init__()

        self.cnn = models.resnet18(pretrained=True)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.unsqueeze(1)

        lstm_out, _ = self.lstm(features)

        output = self.fc(lstm_out[:, -1, :])

        return output
