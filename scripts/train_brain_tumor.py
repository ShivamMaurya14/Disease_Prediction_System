import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings("ignore")

def train_model():
    print("Starting Brain Tumor Model Training...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'tumor-dataset', 'Train')
    test_dir = os.path.join(base_dir, 'tumor-dataset', 'Test')
    model_save_path = os.path.join(base_dir, 'models', 'brain_tumor_model.h5')

    # Data Generators
    batch_size = 32
    img_size = (299, 299)

    _gen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
    ts_gen_datagen = ImageDataGenerator(rescale=1/255)

    print(f"Loading training data from: {train_dir}")
    tr_gen = _gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    
    print(f"Loading testing data from: {test_dir}")
    ts_gen = ts_gen_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=16, class_mode='categorical', shuffle=False)

    # Model Definition (Xception)
    print("Building Xception-based model...")
    img_shape = (299, 299, 3)
    base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax') # 4 classes: glioma, meningioma, notumor, pituitary
    ])

    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    epochs = 5 # Reduced epochs for quick demonstration/setup. Increase for better accuracy.
    print(f"Training for {epochs} epochs...")
    history = model.fit(tr_gen, epochs=epochs, validation_data=ts_gen, shuffle=False)

    # Save Model
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
