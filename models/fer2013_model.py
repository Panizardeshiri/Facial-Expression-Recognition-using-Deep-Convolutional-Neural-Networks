import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout

def build_fer2013_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Proposed Custom Architecture for the FER2013 Dataset
    Source: Master's Thesis - Section 2-3-2
    Target Accuracy: 73.61%
    """
    inputs = Input(shape=input_shape)
    
    # --- BLOCK 1: Dual Extraction Layers ---
    x = Conv2D(256, (3, 3), padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization(name='block1_bn1')(x)
    x = Activation('relu', name='block1_relu1')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_bn2')(x)
    x = Activation('relu', name='block1_relu2')(x)
    x = MaxPool2D((2, 2), name='block1_pool')(x)
    
    # --- BLOCK 2 ---
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv')(x)
    x = BatchNormalization(name='block2_bn')(x)
    x = Activation('relu', name='block2_relu')(x)
    x = MaxPool2D((2, 2), name='block2_pool')(x)
    
    # --- BLOCK 3 ---
    x = Conv2D(64, (3, 3), padding='same', name='block3_conv')(x)
    x = BatchNormalization(name='block3_bn')(x)
    x = Activation('relu', name='block3_relu')(x)
    x = MaxPool2D((2, 2), name='block3_pool')(x)
    
    # --- BLOCK 4 ---
    x = Conv2D(64, (3, 3), padding='same', name='block4_conv')(x)
    x = BatchNormalization(name='block4_bn')(x)
    x = Activation('relu', name='block4_relu')(x)
    x = MaxPool2D((2, 2), name='block4_pool')(x)
    
    # --- DEEP MULTI-STAGE FULLY CONNECTED LAYERS ---
    x = Flatten(name='flatten')(x)
    
    # FC Stage 1 (256 units)
    x = Dense(256, name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Activation('relu', name='fc1_relu')(x)
    x = Dropout(0.5, name='fc1_dropout')(x)
    
    # FC Stage 2 (128 units)
    x = Dense(128, name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Activation('relu', name='fc2_relu')(x)
    x = Dropout(0.5, name='fc2_dropout')(x)
    
    # FC Stage 3 (64 units)
    x = Dense(64, name='fc3')(x)
    x = BatchNormalization(name='fc3_bn')(x)
    x = Activation('relu', name='fc3_relu')(x)
    x = Dropout(0.5, name='fc3_dropout')(x)
    
    # Output projection layer
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Thesis_FER2013_Architecture")
    return model