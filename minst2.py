#https://stackoverflow.com/questions/53131429/valueerror-error-when-checking-input-expected-flatten-input-to-have-3-dimensio

import tensorflow as tf
import numpy as np
import os.path
import matplotlib.pyplot as plt


MODEL_FILENAME="mnist2_modell"

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


if( os.path.exists(MODEL_FILENAME)):
    print("Loading model...")
    model=tf.keras.models.load_model(MODEL_FILENAME)
else:

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(64, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Training model...")
    model.fit(x_train, y_train, epochs=20)
    print("Saving model...")
    model.save(MODEL_FILENAME)

print("Nehany adat a modellrol")
print("A modell pontossaga")
model.evaluate(x_test, y_test)
print("A modell leirasa")
model.summary()
# itt most a predict az azt is szeretné tudni, hogy hány darab képról kell jósolnia.
# Szóval egy 1x 28x 28-as listát adunk be neki. (nem csak 1 ről tudna jósolni)

myprediction = model.predict( x_test[4444].reshape(1,28,28))
print("Szerintem ez egy {}".format(np.argmax(myprediction[0])))
plt.imshow(x_test[4444].reshape(28,28), cmap='Greys')
plt.show()

myprediction = model.predict( x_test[5555].reshape(1,28,28))
print("Szerintem ez egy {}".format(np.argmax(myprediction[0])))
plt.imshow(x_test[5555].reshape(28,28), cmap='Greys')
plt.show()

myprediction = model.predict( x_test[9999].reshape(1,28,28))
print("Szerintem ez egy {}".format(np.argmax(myprediction[0])))
plt.imshow(x_test[9999].reshape(28,28), cmap='Greys')
plt.show()
