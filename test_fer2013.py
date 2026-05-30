import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = './fer2013/test'

if __name__ == '__main__':
    if not os.path.exists('Fer.h5'):
        print("Weights file 'Fer.h5' missing.")
    else:
        categories = sorted(os.listdir(TEST_DIR))
        test_images, test_labels = [], []
        
        for class_idx, category in enumerate(categories):
            class_path = os.path.join(TEST_DIR, category)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    test_images.append(cv2.resize(img, (48, 48)).reshape(48, 48, 1))
                    test_labels.append(class_idx)
                    
        X_test = np.array(test_images) / 255.0
        Y_true = np.array(test_labels)
        
        # One-hot encode the true labels so Keras can calculate the evaluation loss correctly
        Y_true_categorical = tf.keras.utils.to_categorical(Y_true, num_classes=7)
        
        print("Loading trained weights file 'Fer.h5'...")
        model = tf.keras.models.load_model('Fer.h5')
        # --- ADD THIS LINE TO FIX THE METRICS ERROR COMPLETELY ---
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # --------------------------------------------------------
        
        # 1. Corrected Evaluation Line (Using true categorical data)
        # loss, accuracy = model.evaluate(X_test, Y_true_categorical, verbose=0)
        
        # 1. Corrected Evaluation Line (Using true categorical data)
        loss, accuracy = model.evaluate(X_test, Y_true_categorical, verbose=0)
        print(f"\n==============================================")
        print(f"  Verified Test Classification Accuracy: {accuracy * 100:.2f}%")
        print(f"==============================================")
        
        # 2. Generate Predictions for reports
        print("\nRunning inference engine on test data split...")
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        print("\n📊 FER2013 Evaluation Metrics:")
        print(classification_report(Y_true, y_pred, target_names=categories))
        
        print("\n🧩 Confusion Matrix:\n", confusion_matrix(Y_true, y_pred))