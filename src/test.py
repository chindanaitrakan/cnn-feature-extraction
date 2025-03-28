import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.dataset import CIFAR10Data
from model.cnn import CNNModel



# # PCA Visualization
# def log_pca(writer, images, labels, epoch, tag='pca'):
#     images = images.view(images.size(0), -1)  # Flatten the images
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(images.detach().cpu().numpy())

#     writer.add_scalar(f'{tag}/pca_x', pca_result[:, 0], epoch)
#     writer.add_scalar(f'{tag}/pca_y', pca_result[:, 1], epoch)

# # Logging activations
# def log_activations(writer, model, images, epoch):
#     def save_activation_hook(layer_name):
#         def hook(model, input, output):
#             writer.add_histogram(f'{layer_name}/activation', output, epoch)
#         return hook

#     # Hook for the first and last CNN layers
#     hook1 = model.conv1.register_forward_hook(save_activation_hook("conv1"))
#     hook2 = model.conv3.register_forward_hook(save_activation_hook("conv3"))

#     # Forward pass
#     model(images)

#     # Remove the hooks
#     hook1.remove()
#     hook2.remove()

# # Save 16 filter images from the first layer (conv1)
# def save_filter_images(writer, model, epoch):
#     filters = model.conv1.weight.data.clone().detach()
#     grid = torchvision.utils.make_grid(filters[:16], nrow=4)
#     writer.add_image(f'conv1_filters', grid, epoch)

# Training Loop
def train(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    writer = SummaryWriter(log_dir='./runs/cifar10_experiment')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Log metrics to TensorBoard
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/accuracy', epoch_accuracy, epoch)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        writer.add_scalar('test/accuracy', test_accuracy, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    writer.close()

if __name__ == "__main__":
    # Initialize data loaders
    train_dataset = CIFAR10Data(train=True)
    test_dataset = CIFAR10Data(train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = CNNModel(input_shape=(32, 32, 1), num_classes=10)
    import torchinfo
    torchinfo.summary(model, input_size=(1, 1, 32, 32)) 

    # Training loop
    train(model, train_loader, test_loader, num_epochs=50)
