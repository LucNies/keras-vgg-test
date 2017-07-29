import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

from IPython import embed
import config
from prepare_data import load_train_set

def create_network():

    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu', input_shape=config.IMAGE_SHAPE))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def train():

    print("Creating network...")
    model = create_network()
    print('Loading data...')
    data, labels, mean = load_train_set()
    print("Training network...")
    labels = keras.utils.to_categorical(labels, num_classes=config.N_CLASSES)
    model.fit(data, labels, batch_size=64, epochs=20)
    score = model.evaluate(data, labels, batch_size=32)
    embed()
    print("Training done")


if __name__ == "__main__":
    train()