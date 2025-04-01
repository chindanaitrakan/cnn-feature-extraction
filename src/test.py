import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from data.dataset import CIFAR10Data
from model.cnn import CNNModel

if __name__ == "__main__":

    test_dataset = CIFAR10Data(train=False)
    # Tested image which is not involved in training process
    eval_image = test_dataset.eval_image.unsqueeze(0)
    eval_label = test_dataset.eval_label

    # Initialize model
    model = CNNModel(input_shape=(1, 32, 32), num_classes=10)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth", weights_only=True))

    writer = SummaryWriter(log_dir='./runs/cifar10_experiment/feats/')
    writer.add_image('original_image', eval_image[0], 0)

    activations = {}
    def getActivation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    h1 = model.encoder[1].register_forward_hook(getActivation('encoder1_conv2d'))
    h2 = model.encoder[3].register_forward_hook(getActivation('encoder3_conv2d'))
    h3 = model.encoder[6].register_forward_hook(getActivation('encoder6_conv2d'))
    h4 = model.encoder[8].register_forward_hook(getActivation('encoder8_conv2d'))
    h5 = model.encoder[11].register_forward_hook(getActivation('encoder11_conv2d'))
    h6 = model.encoder[13].register_forward_hook(getActivation('encoder13_conv2d'))

    output = model(eval_image)

    for name in activations:
        print(activations[name].shape)
        grid = make_grid(activations[name][0].unsqueeze(1), nrow=4)
        writer.add_image(name, grid, 0)
    writer.close()

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    h6.remove()
    