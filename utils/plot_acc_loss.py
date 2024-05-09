import matplotlib.pyplot as plt
import os

sys_path = os.getcwd()

class Plot:
    def __init__(self, train_acc_history, train_loss_history):
        self.train_acc_history = train_acc_history
        self.train_loss_history = train_loss_history

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.train_acc_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss and Accuracy')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(sys_path, 'plots/accuracy_loss.png')
        plt.savefig(save_path)
        print(f'Plots saved to {save_path}')
        # plt.show()

