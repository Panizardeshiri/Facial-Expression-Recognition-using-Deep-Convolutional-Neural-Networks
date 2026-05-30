import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout

def build_ckplus_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Proposed Custom Architecture for the CK+ Dataset
    Source: Master's Thesis - Section 2-3-1
    Target Accuracy: 100.00%
    """
    inputs = Input(shape=input_shape)
    
    # --- BLOCK 1 ---
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv')(inputs)
    x = Activation('relu', name='block1_relu')(x)
    x = MaxPool2D((2, 2), name='block1_pool')(x)
    
    # --- BLOCK 2 ---
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv')(x)
    x = Activation('relu', name='block2_relu')(x)
    x = MaxPool2D((2, 2), name='block2_pool')(x)
    
    # --- BLOCK 3 ---
    x = Conv2D(64, (3, 3), padding='same', name='block3_conv')(x)
    x = Activation('relu', name='block3_relu')(x)
    x = MaxPool2D((2, 2), name='block3_pool')(x)
    
    # --- DENSE CLASSIFICATION LAYERS ---
    x = Flatten(name='flatten')(x)
    
    x = Dense(128, name='fc1')(x)
    x = Activation('relu', name='fc1_relu')(x)
    x = Dropout(0.4, name='fc1_dropout')(x)  # Exact 0.4 drop rate matching thesis
    
    # Output projection layer
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Thesis_CKPlus_Architecture")
    return model