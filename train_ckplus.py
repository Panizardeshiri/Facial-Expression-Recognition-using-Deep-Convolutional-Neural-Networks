import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.ck_model import build_ckplus_model

# Points to your exact lowercase folder name from the screenshot
DATADIR = './ck+'

def load_ckplus_images():
    categories = sorted([d for d in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, d))])
    images, labels = [], []
    for class_idx, category in enumerate(categories):
        class_path = os.path.join(DATADIR, category)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (48, 48))
                images.append(img_resized.reshape(48, 48, 1))
                labels.append(class_idx)
    return np.array(images) / 255.0, tf.keras.utils.to_categorical(labels, num_classes=7)

if __name__ == '__main__':
    if not os.path.exists(DATADIR):
        print(f"Directory '{DATADIR}' not found.")
    else:
        X, Y = load_ckplus_images()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0, shuffle=True)
        
        model = build_ckplus_model()
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint('BestCK+.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=8, callbacks=callbacks)