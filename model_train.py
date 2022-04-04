import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

## TRAIN PATH
DATA_PATH = './data/'
IMG_SIZE = (224, 244)

## Model Configuration
BATCH_SIZE = 8
EPOCHS = 20

## Use the MobileNet model for object detection and discards the last 1000 Neuron Layer.
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
## Final Layer Output, 3 Classes = [Green, Overripe, Ripe]
output_prediction = Dense(3, activation='softmax')(x)

## Creating Own Model from the base model®
model = Model(inputs=base_model.input, outputs=output_prediction)
## Traning the Train Dataset using Image Generator
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True)


val_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True)


## Compiling the model


model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

callbacks = [ModelCheckpoint('model_chkpt/weights.{epoch:02d}_{val_loss:.4f}_{val_accuracy:.4f}.h5')]

# history = model.fit_generator(
#     generator=train_generator,
#     steps_per_epoch= train_generator.samples // BATCH_SIZE,
#     validation_data= val_generator,
#     validation_steps= val_generator.samples // BATCH_SIZE,
#     callbacks= callbacks,
#     epochs=EPOCHS
# )
#
#
# model.save('models/model_2.h5')
#
# print("PRINGTING HISTORY")
# print(history)
#

