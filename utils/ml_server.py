import numpy as np
import tensorflow as tf
import json
from flask import Flask, request

mnist_model = tf.keras.models.load_model("/utils/model/mnist_model.h5")
feature_mnist_model = tf.keras.models.Model(mnist_model.inputs,
                                            [layer.output for layer in mnist_model.layers])


_, (X_test, _) = tf.keras.datasets.mnist.load_data()
X_test = X_test / 255.

def get_prediction():
  index = np.random.choice(X_test.shape[0])
  image = X_test[index, :, :]
  image_arr = np.reshape(image, (1, 784))
  return feature_mnist_model.predict(image_arr), image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def model():
  if request.method == 'POST':
    predicts, img = get_prediction()
    final_preds = [pred.tolist() for pred in predicts]

    return json.dumps({
        "predictions": final_preds,
        "image": img.tolist()
    })
  return '<h1 align="center">MODEL DEPLOYED SUCCESSFULLY</h1>'

if __name__ == '__main__':
  app.run()