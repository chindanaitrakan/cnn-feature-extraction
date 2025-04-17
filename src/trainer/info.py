import torch
import numpy as np
from sklearn.metrics import mutual_info_score
from data.dataset import CIFAR10Data
from model.cnn import CNNModel
from trainer.utils import *
from torch.utils.data import DataLoader


def compute_mutual_info(x, y, bins=255):
    x = x.flatten()
    y = y.flatten()
    
    x_discrete = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_discrete = np.digitize(y, np.histogram_bin_edges(y, bins=bins))

    return mutual_info_score(x_discrete, y_discrete)

def average_mutual_information(images, activations, bins=255):
    """
    Compute average mutual information between images and activations, taking channel differences into account.

    Args:
        images: Tensor of shape (b, c1, h1, w1)
        activations: Tensor of shape (b, c2, h2, w2)
        bins: Number of bins for MI histogram
        max_pairs: Maximum number of channel pairs to compute (for speed)

    Returns:
        avg_mi: Average mutual information across batch and channel pairs
    """
    b, c1, h1, w1 = images.shape
    _, c2, h2, w2 = activations.shape

    # Resize activation maps to match image size
    if (h1, w1) != (h2, w2):
        activations = torch.nn.functional.interpolate(activations, size=(h1, w1), mode='bilinear', align_corners=False)

    total_mi = 0.0
    total_pairs = 0
    for i in range(b):
        img_sample = images[i].cpu().numpy()
        act_sample = activations[i].cpu().numpy()

        #average over channels
        for ci in range(c1):
            for cj in range(c2):
                mi = compute_mutual_info(img_sample[ci], act_sample[cj], bins=bins)
                total_mi += mi
                total_pairs += 1

    return total_mi / total_pairs if total_pairs > 0 else 0.0

if __name__ == "__main__":

    train_dataset = CIFAR10Data(train=True)
    train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True)

    train_data, train_labels = next(iter(train_loader))

    # Initialize model
    model = CNNModel(input_shape=(1, 32, 32), num_classes=10)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth", weights_only=True))

    # # Print all direct children modules with their names
    # for name, child in model.named_children():
    #     print(f"Name: {name}, Module: {child}")

    activations = {}
    def getActivation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    h = model.encoder[13].register_forward_hook(getActivation('encoder13_conv2d'))

    output = model(train_data)
    activation = activations['encoder13_conv2d']

    print(train_data.shape)
    print(activation.shape)

    mi = average_mutual_information(train_data[:2], activation)
    print(mi)

    h.remove()

