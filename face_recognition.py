import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
dataset_path = "classmates_faces"
class_names = os.listdir(dataset_path)

# Prepare the data
def load_images():
    images = []
    labels = []
    for label, class_name in enumerate(class_names):
        image_path = os.path.join(dataset_path, class_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))  # Resize images to a consistent size
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_images()
X = X.astype('float32') / 255.0  # Normalize pixel values

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model using data augmentation
model.fit(datagen.flow(X, y, batch_size=32), epochs=50)

# Save the model
model.save("face_recognition_model.h5")
