import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.fer2013_model import build_fer2013_model

TRAIN_DIR = './fer2013/train'

def load_and_balance_fer2013():
    categories = sorted(os.listdir(TRAIN_DIR))
    raw_images, raw_labels = [], []
    
    for class_idx, category in enumerate(categories):
        class_path = os.path.join(TRAIN_DIR, category)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                raw_images.append(cv2.resize(img, (48, 48)).flatten())
                raw_labels.append(class_idx)
                
    # Re-balancing configurations matching your thesis notebooks
    oversampler = RandomOverSampler(sampling_strategy={0:6000, 1:6000, 2:6500, 3:9500, 4:6700, 5:6000, 6:6700})
    X_over, Y_over = oversampler.fit_resample(np.array(raw_images), np.array(raw_labels))
    
    undersampler = RandomUnderSampler(sampling_strategy={0:4000, 1:5000, 2:6000, 3:7000, 4:6000, 5:4100, 6:6200})
    X_resampled, Y_resampled = undersampler.fit_resample(X_over, Y_over)
    
    X_balanced = X_resampled.reshape(-1, 48, 48, 1) / 255.0
    Y_balanced = tf.keras.utils.to_categorical(Y_resampled, num_classes=7)
    return X_balanced, Y_balanced

if __name__ == '__main__':
    if not os.path.exists(TRAIN_DIR):
        print("FER2013 training directory not found.")
    else:
        X_train, Y_train = load_and_balance_fer2013()
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, shuffle=True)
        
        datagen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.1,
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
        )
        datagen.fit(X_train)
        
        model = build_fer2013_model()
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
            ModelCheckpoint('Fer.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        model.fit(datagen.flow(X_train, Y_train, batch_size=64), validation_data=(X_val, Y_val), epochs=80, callbacks=callbacks)