import tensorflow as tf

x=(-2, -3, 1, 2, 3, 4, 5)
#y = 2*x+1
y=(-3, -5, 3, 5, 7, 9,11)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape=(1,))
])

model.summary()

# SGD: square gradient desent
# loss: hibanegyzet osszegek
model.compile(optimizer='SGD',
              loss='mean_squared_error',
              )

print("Trying to learn")
print("Check how the loss is getting smaller and smaller")
model.fit(x,y,epochs=1000)

print("Trying to predict")
xtest=(100,200,300,500)

print(xtest, model.predict(xtest))

print(model.get_weights())

