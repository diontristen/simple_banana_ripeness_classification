import tensorflowjs as tfjs
import tensorflow as tf



TEST_PATH = './test'
IMG_SIZE = (224, 244)
model = tf.keras.models.load_model('./models/model_1.h5')
tfjs.converters.save_keras_model(model, 'model_output/')