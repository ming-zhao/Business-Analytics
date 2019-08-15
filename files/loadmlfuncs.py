import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import *
from IPython.display import display, HTML
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

dataurl = 'https://raw.githubusercontent.com/ming-zhao/Business-Analytics/master/data/data_mining/'

def transfer_data(df, df_quantile):
    for col in ['month', 'art_book']:
        df = pd.merge_asof(df.sort_values(col),df_quantile[[col, 'quantile']],
                           on=col, direction='backward').sort_values('customer')\
        .drop(columns=[col]).rename(columns={'quantile': col})
    df = df.set_index('customer')
    return df

def net_compare(nNodes, train_size_pct):
    from keras.datasets import mnist
    from keras import models
    from keras import layers
    from keras.utils import to_categorical
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
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