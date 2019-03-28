import json
import tensorflow as tf
import numpy as np
from azureml.core.model import Model
from keras.models import load_model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('tf_mnist_model')
    model = load_model(model_path, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict_classes(data)
    return json.dumps(y_hat.tolist())
