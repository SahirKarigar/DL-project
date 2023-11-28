import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import transform
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# Load and preprocess the images
real_path = "Humans"
ai_path = "AI"

real_images = []
ai_images = []
for filename in os.listdir(real_path):
    img = keras.preprocessing.image.load_img(
        os.path.join(real_path, filename), target_size=(128, 128)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    real_images.append(img)

for filename in os.listdir(ai_path):
    img = keras.preprocessing.image.load_img(
        os.path.join(ai_path, filename), target_size=(128, 128)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    ai_images.append(img)

# Split the data into training and testing sets
X = np.vstack((real_images, ai_images))
y = np.hstack((np.zeros(len(real_images)), np.ones(len(ai_images))))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

batch_size = 32
total_batches = len(X_train) // batch_size

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10,batch_size=batch_size, validation_split=0.2,steps_per_epoch=total_batches)

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

train_loss, train_acc = model.evaluate(X_train, y_train)
print('train accuracy:',train_acc  )
model.save("classifier5.h5")

print('done')



def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (128, 128, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

new_image = load('na.png')
prediction = model.predict(new_image)
if prediction > 0.5:
    print('AI-generated portrait')
else:
    print('Real portrait')

new_image = load('aaa.png')
prediction = model.predict(new_image)
if prediction > 0.5:
    print('AI-generated portrait')
else:
    print('Real portrait')

new_image = load('hu.png')
prediction = model.predict(new_image)
if prediction > 0.5:
    print('AI-generated portrait')
else:
    print('Real portrait')

new_image = load('huma.png')
prediction = model.predict(new_image)
if prediction > 0.5:
    print('AI-generated portrait')
else:
    print('Real portrait')
