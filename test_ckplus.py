import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATADIR = './ck+'

if __name__ == '__main__':
    if not os.path.exists('BestCK+.h5'):
        print("Weights file 'BestCK+.h5' missing.")
    else:
        categories = sorted([d for d in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, d))])
        images, labels = [], []
        for class_idx, category in enumerate(categories):
            class_path = os.path.join(DATADIR, category)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, (48, 48)).reshape(48, 48, 1))
                    labels.append(class_idx)
                    
        X = np.array(images) / 255.0
        Y = tf.keras.utils.to_categorical(labels, num_classes=7)
        _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
        
        model = tf.keras.models.load_model('BestCK+.h5')
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(Y_test, axis=1)
        
        # Evaluate basic accuracy
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        print(f"\n==============================================")
        print(f"  Verified Test Classification Accuracy: {accuracy * 100:.2f}%")
        print(f"==============================================")
        
        print("\n📊 CK+ Evaluation Metrics:")
        print(classification_report(y_true, y_pred, target_names=categories))
        print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))