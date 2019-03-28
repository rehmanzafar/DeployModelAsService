import requests
import tensorflow as tf
mnist = tf.keras.datasets.mnist

n_channels = 1

img_rows, img_cols = 28, 28

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
input_shape = (img_rows, img_cols, n_channels)

input_data = "{\"data\": " + str(x_test[0:5,].tolist()) + "}"

headers = {'Content-Type':'application/json'}

use_your_uri = "Use your URI here"
resp = requests.post(use_your_uri, input_data, headers=headers)

print("prediction:", resp.text)
