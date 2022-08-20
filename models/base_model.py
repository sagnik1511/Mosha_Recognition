import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential

def BaseModel(model_config):

    model = Sequential()
    for layer in model_config:
        # Input Layer
        if layer[0] == "input":
            model.add(Input(layer[1]))
        # Convolution Layer
        elif layer[0] == "conv":
            model.add(Conv2D(filters=layer[1][0], kernel_size=layer[1][1],
            strides=layer[1][2], padding=layer[1][3], activation=layer[1][4]))
        # MaxPool Layer
        elif layer[0] == "maxpool":
            model.add(MaxPool2D(2))
        # Dropout Layer
        elif layer[0] == "dropout":
            model.add(Dropout(layer[1][0]))
        # Flatten layer
        elif layer[0] == "flatten":
            model.add(Flatten())
        # Lienar / Dense Layer
        elif layer[0] == "dense":
            model.add(Dense(units=layer[1][0], activation=layer[1][1]))
        else:
            raise NotImplementedError

    return model
