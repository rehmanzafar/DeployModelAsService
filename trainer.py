import tensorflow as tf
import numpy as np
from keras.models import load_model

mnist = tf.keras.datasets.mnist

batch_size = 512
num_classes = 10
epochs = 1
n_channels = 1

img_rows, img_cols = 28, 28

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
input_shape = (img_rows, img_cols, n_channels)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3),
                 activation='relu',
                 input_shape=input_shape),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(model.summary())
pred = model.predict(np.expand_dims(x_test[0], axis=0))

model.save('tf_mnist_model.h5')

l_model = load_model('tf_mnist_model.h5', custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})
print(l_model.summary())
pred = l_model.predict_classes(np.expand_dims(x_test[0], axis=0))
print('done')
