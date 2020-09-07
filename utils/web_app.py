import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import requests

uri = 'http://127.0.0.1:5000'
st.markdown("# Handwritten Digits (MNIST) Visualizer")
if st.button('Get Prediction'):
  resp = requests.post(uri, data={})
  resp = json.loads(resp.text)
  st.markdown('## Random Input Image')
  predicts = resp.get('predictions')
  image = resp.get('image')
  image = np.reshape(image, (28, 28))

  st.image(image, width=150)

  st.markdown('## Model Visualizer')

  for layers, pred in enumerate(predicts):
    nums = np.squeeze(np.array(pred))

    plt.figure(figsize=(32, 4))

    if layers == 2:
      row, col = 1, 10
    else:
      row, col = 2, 16
    
    for i, num in enumerate(nums):
      plt.subplot(row, col, i+1)
      plt.imshow(num * np.ones((8, 8, 3)).astype('float32'))
      plt.xticks([])
      plt.yticks([])

      if layers == 2:
        plt.xlabel(str(i), fontsize=50)
    plt.subplots_adjust(hspace=1, wspace=1)
    plt.tight_layout()

    st.text('Layer {} with {} units'.format(layers+1, i+1))
    st.pyplot()