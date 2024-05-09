import torch


class Eval:
    def __init__(self, model, test_loader, criterion, device, encode_labels):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.encode_labels = encode_labels
        self.k_v_pair = dict()

    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient tracking
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), torch.tensor(labels).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                self.k_v_pair[self.encode_labels.get(str(predicted))] = self.encode_labels.get(labels)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        print('\nLabel\t|\tPredictions\n')
        for item in self.k_v_pair:
            print(f'{item[1]}\t|\t{item[0]}')
