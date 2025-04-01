import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import io
import torchvision.transforms as transforms
import torchvision.io
import torch

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image_bytes = torch.tensor(list(buf.getvalue()), dtype=torch.uint8)
    image = torchvision.io.decode_png(image_bytes)
    image = image.float()/255.

    return image.unsqueeze(0)

def get_confusion_matrix(y_labels, preds, class_names):
    preds = preds.numpy()
    cm = sklearn.metrics.confusion_matrix(
        y_labels, preds, labels=np.arange(len(class_names)),
    )

    return cm


def plot_confusion_matrix(cm, class_names):
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    # Normalize Confusion Matrix
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3,)

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                i, j, cm[i, j], horizontalalignment="center", color=color,
            )

    plt.tight_layout()
    plt.xlabel("True Label")
    plt.ylabel("Predicted label")

    cm_image = plot_to_image(figure)
    return cm_image
