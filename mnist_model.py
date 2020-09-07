# %%
## Import libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# %%
## Download dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# %%
## Plot the dataset
width, height = 10, 10

fig, ax = plt.subplots(height, width, figsize=(10, 10))
ax = ax.ravel()

n_train = len(X_train)

for i in np.arange(0, width, height):
    index = np.random.randint(0, n_train)
    ax[i].imshow(X_train[index], cmap='binary')
    ax[i].set_title(y_train[index], fontsize=15)
    ax[i].axis('off')

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)


# %%
## Normalize the data
X_train = np.reshape(X_train, (60000, 28*28))
X_test = np.reshape(X_test, (10000, 28*28))

X_train, X_test = X_train/255. , X_test/255.



# %%
## Build the model
mnist_model = keras.Sequential([
                                keras.layers.Dense(units=32,
                                                   activation='sigmoid',
                                                   input_shape=(28*28,)),
                                keras.layers.Dense(units=32,
                                                   activation='sigmoid'),
                                keras.layers.Dense(units=10,
                                                   activation='softmax')

])

mnist_model.summary()


# %%
## Compile the Model
mnist_model.compile(optimizer='adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                    )



# %%
## Train the model with data
hist = mnist_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                       epochs=100,
                       batch_size=1024)



# %%
# save the model
mnist_model.save('./utils/model/mnist_model.h5')
