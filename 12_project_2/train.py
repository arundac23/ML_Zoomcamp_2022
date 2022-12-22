#!/usr/bin/env python
# coding: utf-8

# unzip intel_image_classification.zip')
# # !rm intel_image_classification.zip


import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

import tensorflow.keras.applications.xception as xc

from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.inception_v3 import InceptionV3


from tensorflow.keras.applications.xception import preprocess_input


path = 'intel_image_classification/train/buildings'
name = '47.jpg'
fullname = f'{path}/{name}'
load_img(fullname)

#original dataset is available in kaggle 'https://www.kaggle.com/datasets/puneet6060/intel-image-classification?select=seg_train'

train_dir = Path('intel_image_classification/train')
train_filepaths = list(train_dir.glob(r'**/*.jpg'))

val_dir = Path('intel_image_classification/val')
val_filepaths = list(val_dir.glob(r'**/*.jpg'))

test_dir = Path('intel_image_classification/test')
test_filepaths = list(test_dir.glob(r'**/*.jpg'))


def proc_img(filepath):

    labels = [str(filepath[i]).split("/")[-2]               for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='File_Location').astype(str)
    labels = pd.Series(labels, name='Category_Name')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df

train_df = proc_img(train_filepaths)
val_df = proc_img(val_filepaths)
full_train_df = pd.concat([train_df, val_df])


def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

input_size = 299


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_ds = train_gen.flow_from_directory(
    'intel_image_classification/train',
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    'intel_image_classification/val',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    #'xception_v_1_best_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


learning_rate = 0.0005
size = 100
droprate = 0.5

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    'intel_image_classification/test',
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

X_model = keras.models.load_model('xception_v_1_11_0.930.h5')

X_model.evaluate(test_ds)

converter = tf.lite.TFLiteConverter.from_keras_model(X_model)

model_lite = converter.convert()

with open('xception_model.tflite', 'wb') as f_out:
    f_out.write(model_lite)

