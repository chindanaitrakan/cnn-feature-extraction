import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from .info import average_mutual_information
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_confusion_matrix, plot_confusion_matrix

class_names = [
    "Airplane",
    "Autmobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            torch.save(self.best_model_state, "./model/checkpoint/best_model.pth") 
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


class Trainer:

    def __init__(self, model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eps = num_epochs
        self.lr = learning_rate
        self.example_data, self.targets = next(iter(test_loader))
        model.apply(self.initialize_weights)

        self.writer = SummaryWriter(log_dir='./runs/cifar10_experiment')
        # Perform feature projection on example data
        # select random images and their target indices
        images, labels = self.select_n_random(self.example_data, self.targets)

        # get the class labels for each image
        class_labels = [class_names[lab] for lab in labels]

        # log embeddings
        features = images.view(-1, 32 * 32)
        self.writer.add_embedding(features,
                            metadata=class_labels,
                            label_img=images)
        
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight) 
            if m.bias is not None:
                init.zeros_(m.bias) 
    
    # helper function
    def select_n_random(self, data, labels, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]
            
    def train(self):

        # write computational graph
        self.writer.add_graph(self.model, self.example_data)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-7)
        early_stopping = EarlyStopping(patience=5, delta=0.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device {device}")
        self.model.to(device)

        # define hook to extract last CNN activation
        activations = {}
        def getActivation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        h = self.model.encoder[13].register_forward_hook(getActivation('encoder13_conv2d'))
        mi_list = []

        for epoch in range(self.eps):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            confusion = np.zeros((len(class_names), len(class_names)))

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                confusion += get_confusion_matrix(labels.cpu(), predicted.cpu(), class_names)
            
            # Log confusion matrics on train set
            cm_tensor = plot_confusion_matrix(confusion / batch_idx, class_names)
            self.writer.add_image("Confusion Matrix", cm_tensor[0], global_step=epoch)

            # Calculate training loss and accuracy
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = correct / total

            self.model.eval()
            correct = 0
            total = 0
            running_loss =0
            with torch.no_grad():
                mi = 0
                count = 0
                for images, labels in self.test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)

                    test_loss = criterion(outputs, labels)
                    running_loss += test_loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    mi += average_mutual_information(images, activations['encoder13_conv2d'])
                    count += 1
                mi_list.append(mi/count)

            test_loss = running_loss / len(self.test_loader)
            test_accuracy = correct / total

            #Log losses and accuracies
            self.writer.add_scalars('Loss', {'train':train_loss,
                                            'test':test_loss,}, epoch)
            self.writer.add_scalars('Accuracy', {'train':train_accuracy,
                                            'test':test_accuracy,}, epoch)
            

            print(f"Epoch {epoch+1}/{self.eps}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            early_stopping(test_accuracy, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # Generate a list of epoch numbers starting from 1
        epochs = list(range(1, len(mi_list) + 1))

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, mi_list, marker='o', linestyle='-', color='blue', label='Mutual Information')
        plt.title('Mutual Information Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mutual Information')
        plt.grid(True)
        plt.legend()

        # Save the plot as a PNG image
        plt.savefig('../outputs/mutual_information_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.writer.close()
        h.remove()





