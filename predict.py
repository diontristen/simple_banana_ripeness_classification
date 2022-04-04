import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import tensorflow as tf
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec




TEST_PATH = './test'
IMG_SIZE = (224, 244)
model = tf.keras.models.load_model('./models/model_2.h5')

def prepare_image(number):
    image = cv2.imread(f"{TEST_PATH}/{number}.jpg")[...,::-1]
    pred_images = cv2.resize(image, IMG_SIZE)
    pred_images = np.array([pred_images])
    print(pred_images.shape)
    return pred_images, image


test_number = 1
while os.path.isfile(f"{TEST_PATH}/{test_number}.jpg"):
    try:
        image, originalImage = prepare_image(test_number)
        prediction = model.predict(image)
        idx_prediction = np.argmax(prediction[0])
        print(idx_prediction, prediction[0])
        plot.imshow(originalImage)
        plot.show()
    except Exception as e:
        print("Something went wrong with the image...")
        print(e)
    finally:
        test_number += 1

