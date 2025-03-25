import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

from src.data.data import Data
from src.model.cnn import get_new_model
from src.model.utils import *

def save_metrics_plot(history, name, file="./outputs/metrics/"):

    save = file+name+".png"
    history = pd.DataFrame(history.history)
    fig, axs = plt.subplots(1,2)

    axs[0].plot(history['loss'])
    axs[0].plot(history['val_loss'])
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].set_title(name+'losses')

    axs[1].plot(history['accuracy'])
    axs[1].plot(history['val_accuracy'])
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].set_title(name+'accuracies')

    plt.tight_layout()
    plt.savefig(save, dpi=300)

if __name__ == "__main__":

    cifar_dataset = Data()
    (train_data, train_labels_encoded), (test_data, test_labels_encoded) = cifar_dataset.all()
    print(train_data.shape)
    
    # training model on larger dataset 
    model_50k = get_new_model(input_shape=(32,32,1))

    saving_direc = "./outputs/checkpoints/"
    checkpoint_best_epoch = get_checkpoint_best_epoch(f'{saving_direc}best_epoch_50k.weights.h5')
    early_stopping = get_early_stopping()

    callbacks = [checkpoint_best_epoch, early_stopping]

    history_50k = model_50k.fit(x=train_data,
                                y=train_labels_encoded,
                                epochs=50,batch_size=128,
                                validation_data=(test_data,test_labels_encoded),
                                callbacks=callbacks)
    
    save_metrics_plot(history_50k, "model_50k")

    # training model on smaller dataset 
    train_data_small = test_data
    train_labels_encoded_small = test_labels_encoded

    test_data_small = train_data
    test_labels_encoded_small = train_labels_encoded

    model_10k =  get_new_model(input_shape=(32,32,1))
    saving_direc = "./outputs/checkpoints/"
    checkpoint_best_epoch = get_checkpoint_best_epoch(f'{saving_direc}best_epoch_10k.weights.h5')
    early_stopping = get_early_stopping()

    callbacks = [checkpoint_best_epoch, early_stopping]


    history_10k = model_10k.fit(x=train_data_small,
                                y=train_labels_encoded_small,
                                epochs=50,batch_size=128,
                                validation_data=(test_data_small,test_labels_encoded_small),
                                callbacks=callbacks)
    
    save_metrics_plot(history_10k, "model_10k")
    





