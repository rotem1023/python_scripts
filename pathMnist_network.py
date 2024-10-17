import pandas as pd
from medmnist import PathMNIST
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Dropout, Dropout, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.layers import BatchNormalization
from keras.layers import Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import torch
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# from medmnist.dataset import PathMNIST
import os


def fine_tune_resnet50(input_shape, num_classes):
    # Load ResNet50 without the top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model
    base_model.trainable = False  # Only fine-tune the newly added layers initially

    # Add new layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Add pooling to reduce the feature dimensions
    x = Dense(256, activation='relu')(x)  # Dense layer with ReLU activation
    x = Dense(num_classes, activation='softmax')(x)  # Output layer for 9 classes

    # Create the final model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def _validate_gpu():
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("Code need to run on GPU")

def extract_logits_and_softmax(model, data):
    # Create a new model that outputs logits
    logits_model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Logits are the output of the second-last layer

    # Get logits
    logits = logits_model.predict(data, batch_size=32)

    # Apply softmax manually to the logits to get the probabilities
    softmax_values = tf.nn.softmax(logits).numpy()  # Convert Tensor to NumPy array
    
    return logits, softmax_values

if __name__ == '__main__':  
    _validate_gpu()
    
    dir_path = './resources/'
    train_dataset = PathMNIST(split="train", download=True)
    test_dataset = PathMNIST(split="test", download=True)
    noisy_labels_train_path= './resources/pathminst_train_noisy_labels.csv'
    noisy_labels_test_path = './resources/pathmnist_test_noisy_labels.csv'
    noisy_labels_test_table=  pd.read_csv(noisy_labels_test_path)
    noisy_labels_train_table = pd.read_csv(noisy_labels_train_path)
    noisy_labels_test = noisy_labels_test_table['Predicted Label'].to_numpy()
    noisy_labels_train = noisy_labels_train_table['Predicted Label'].to_numpy()
    
    x_train = train_dataset.imgs/255
    x_test = test_dataset.imgs/255
    
    
    train_labels = to_categorical(noisy_labels_train,num_classes=9)
    test_labels= to_categorical(noisy_labels_test,num_classes=9)
    
    input_shape = (224, 224, 3)
    x_train_resized = tf.image.resize(x_train, (224, 224))
    x_test_resized = tf.image.resize(x_test, (224, 224))
    # Data Augmentation and Resizing
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    # Flow from numpy array, adjusting size to 224x224 for ResNet50
    train_generator = datagen.flow(x_train_resized, train_labels, batch_size=32)
    val_generator = datagen.flow(x_test_resized, test_labels, batch_size=32)

    model = fine_tune_resnet50(input_shape=input_shape, num_classes=9)
    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        verbose=2  
    )
    
    model.save(f'{dir_path}/pathMnistModel.h5')
    
    # Extract logits and softmax values on the test set
    logits, softmax_values = extract_logits_and_softmax(model, x_test_resized)

    # Save logits and softmax values
    np.save('./resources/test_logits.npy', logits)
    np.save('./resources/test_softmax_values.npy', softmax_values)

    # Predict class probabilities on the test set
    predicted_classes = np.argmax(softmax_values, axis=1)

    # Save the predicted classes to a .npy file
    np.save('./resources/test_predicted_classes.npy', predicted_classes)