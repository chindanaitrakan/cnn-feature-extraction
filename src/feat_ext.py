import matplotlib.pyplot as plt
import numpy as np

from src.model.cnn import *
from src.data.data import Data

if __name__ == "__main__":

    model_50k = get_new_model(input_shape=(32,32,1))
    model_50k.load_weights("./outputs/checkpoints/best_epoch_50k.weights.h5")
    model_10k = get_new_model(input_shape=(32,32,1))
    model_10k.load_weights("./outputs/checkpoints/best_epoch_10k.weights.h5")

    layers_50k = model_50k.layers
    input_50k = model_50k.input

    layer_outputs_50k = [layer.output for layer in layers_50k]
    features_50k = Model(inputs=input_50k, outputs=layer_outputs_50k)

    cifar_dataset = Data()
    eval_image_data = cifar_dataset.eval_data 
    extracted_50k = features_50k(eval_image_data)

    f1_50k = extracted_50k[1]
    imgs = f1_50k[0,...]

    plt.imshow(extracted_50k[0][0], cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(imgs[...,0], cmap='gray')
    plt.axis('off')
    plt.show()

    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck']

    # fig = plt.figure(figsize=(6,4))
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(np.arange(0,10), model_50k.predict(eval_image_data, verbose=False)[0])
    # ax.set_title('Probability distribution', fontsize=20)
    # ax.set_xticks(np.arange(0,10))
    # ax.set_xticklabels(class_names, rotation=45, fontsize=14)
    # ax.set_xlabel('Class name', fontsize=18)
    # ax.set_ylabel('Probability', fontsize=18)
    # plt.show()

