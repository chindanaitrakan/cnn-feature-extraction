from data.dataset import CIFAR10Data
from model.cnn import CNNModel

from trainer.utils import *
from torch.utils.data import DataLoader

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
        activations[name] = output.view(-1, 16*output.shape[2]*output.shape[3]).detach().numpy()
    return hook

# Discretization of continous values from the layers
def discretization(activations_list,bins):

    n_bins = bins
    bins = np.linspace(min(np.min(activations_list, axis=0)),
                               max(np.max(activations_list,axis=0)), n_bins+1)
    activations_list = np.digitize(activations_list, bins)
            
    return activations_list


#h = model.encoder[1].register_forward_hook(getActivation('encoder1_conv2d'))
h2 = model.encoder[13].register_forward_hook(getActivation('encoder13_conv2d'))

output = model(train_data)

#activation1 = discretization(activations['encoder1_conv2d'], 20)
activation2 = discretization(activations['encoder13_conv2d'], 120)
print(activation2.shape)

x = train_data.view(-1, 32*32).numpy()
Info = InfoCalculator()
print(Info.mutual_Info(x, train_labels.numpy()))
print(Info.mutual_Info(x, activation2))
print(Info.mutual_Info(x, x))
#h.remove()
h2.remove()

