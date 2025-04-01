from torch.utils.data import DataLoader

from data.dataset import CIFAR10Data
from model.cnn import CNNModel
from trainer.trainer import Trainer



if __name__ == "__main__":
    # Initialize data loaders
    train_dataset = CIFAR10Data(train=True)
    test_dataset = CIFAR10Data(train=False)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Initialize model
    model = CNNModel(input_shape=(1, 32, 32), num_classes=10)

    # Training loop
    CNNTrainer = Trainer(model, train_loader, test_loader, num_epochs=50)
    CNNTrainer.train()
