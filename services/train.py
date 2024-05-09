import torch
from tqdm import tqdm


class Train:
    def __init__(self, model, optimizer, criterion, train_loader, device, epochs, model_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.model_path = model_path
        self.train_loss_history = []
        self.train_acc_history = []

    def accuracy(self, output, target):
        with torch.no_grad():
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum().item()
            total = target.size(0)
            acc = correct / total
        return acc

    def train(self):
        print('Training...')
        print(f'using {self.device} to train')
        self.model.to(self.device)
        self.model.train()

        for epoch in range(1, self.epochs+1):
            running_loss = 0.0
            running_accuracy = 0.0

            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.epochs}', leave=True) as pbar:
                for images, labels in self.train_loader:
                    images = images.to(self.device)
                    labels = torch.tensor(labels).to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_accuracy += self.accuracy(outputs, labels) * images.size(0)

                    # Update tqdm description with loss and accuracy
                    avg_loss = running_loss / ((pbar.n + 1) * self.train_loader.batch_size)
                    avg_accuracy = running_accuracy / ((pbar.n + 1) * self.train_loader.batch_size)
                    pbar.set_postfix({f'Loss': f'{avg_loss:.4f}', 'Accuracy': f'{avg_accuracy:.4f}'})
                    pbar.update(1)  # Update the progress bar

                epoch_loss = running_loss / len(self.train_loader.dataset)
                epoch_accuracy = running_accuracy / len(self.train_loader.dataset)

                self.train_loss_history.append(epoch_loss)
                self.train_acc_history.append(epoch_accuracy)

            if epoch % 100 == 0:
              torch.save(self.model.state_dict(), self.model_path)
              print(f'Model saved to {self.model_path}')
        return self.train_acc_history, self.train_loss_history
