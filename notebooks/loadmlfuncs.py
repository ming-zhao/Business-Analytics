import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import *
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

dataurl = 'https://raw.githubusercontent.com/ming-zhao/Business-Analytics/master/data/data_mining/'

def net_compare(nNodes, train_size_pct):
    network = models.Sequential()
    network.add(layers.Dense(nNodes, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    train_size = int(60000*train_size_pct)
    
    train_images_reshape = train_images[:train_size].reshape((train_size, 28 * 28))
    train_images_reshape = train_images_reshape.astype('float32') / 255

    test_images_reshape = test_images.reshape((10000, 28 * 28))
    test_images_reshape = test_images_reshape.astype('float32') / 255

    train_labels_cat = to_categorical(train_labels[:train_size])
    test_labels_cat = to_categorical(test_labels)

    network.fit(train_images_reshape, train_labels_cat, epochs=5, batch_size=128);

    test_loss, test_acc = network.evaluate(test_images_reshape, test_labels_cat)
    return test_acc