"""
Hello Dear coders,
This Python script is developed for the purpose of detecting diabetic retinopathy
in retinal fundus photographs. It utilizes a convolutional neural network (CNN)
based on the MobileNetV2 architecture. The code includes functions for loading and
preprocessing image data, generating data augmentations, building and training the
CNN model, and evaluating its performance through accuracy, loss, and a confusion
matrix. The script is structured to first set up the dataset and model configuration,
followed by model training and evaluation. It concludes with visualization of the
model's predictions, providing an understanding of the model's capabilities in
classifying retinal images for the presence of diabetic retinopathy.
***Dataset Kaggle.com***
Python 3.12
Subscribe, Like, Comment and Share please
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from time import perf_counter
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
import os

# Disable optimizations for reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


# Function to load images and create a DataFrame with paths and labels
def load_image_data(image_dir):
    """
    Loads images from a directory into a pandas DataFrame with image file paths
    and corresponding labels. The images are expected to be stored in a hierarchical
    directory structure where each subdirectory corresponds to a label.
    """
    filepaths = list(image_dir.glob(r'**/*.png'))
    labels = [os.path.split(os.path.split(fp)[0])[1] for fp in filepaths]

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    return (pd.concat([filepaths, labels], axis=1).sample(frac=1)
            .reset_index(drop=True))


# Function to display images with labels
def display_images(image_df, RowsNum=3, ColumnsNum=4):
    """
    Displays a grid of images with their corresponding labels for visual inspection.
    This helps to verify the data loading step and ensure that images are correctly
    labeled before training the model.
    """
    fig, axes = plt.subplots(nrows=RowsNum, ncols=ColumnsNum, figsize=(10, 7),
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(image_df.Filepath[i]))
        ax.set_title(image_df.Label[i])
    plt.tight_layout()
    plt.show()


# Function to create data generators
def create_data_generators(dataframe, train_df, test_df):
    """
    Creates image data generators for the training and validation datasets.
    The generators will perform real-time data augmentation to introduce variability
    in the dataset and prevent overfitting.
    """
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training'
    )

    val_images = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation'
    )

    test_images = test_gen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    return train_gen, test_gen, train_images, val_images, test_images


# Function to build a model using MobileNetV2 as a base
def build_model(pretrained_model_func, num_classes):
    """
    Builds a CNN model using MobileNetV2 as the base architecture. The base
    model's weights are frozen to retain the knowledge it gained from pretraining
    on ImageNet. Custom dense layers are added on top to adapt the model to the
    specific task of diabetic retinopathy detection.
    """
    pretrained_model = pretrained_model_func(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet',
        pooling='avg')
    pretrained_model.trainable = False
    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Function to train multiple models and record their performance
def train_models(models, train_images, val_images):
    """
    Trains multiple CNN models using the provided image data generators.
    Training performance is recorded for later analysis and comparison.
    Each model's architecture can be specified individually.
    """
    model_results = []

    for name, model_info in models.items():
        model = build_model(model_info['model_func'], num_classes=5)
        start_time = perf_counter()

        history = model.fit(train_images, validation_data=val_images, epochs=10,
                            verbose=1)
        duration = round(perf_counter() - start_time, 2)

        model_results.append({
            'name': name,
            'model': model,
            'history': history,
            'train_time': duration
        })

    return model_results


# Function to plot training history
def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss over epochs. This
    visual aid is used to monitor the training process and detect signs of
    overfitting or underfitting.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# Function to evaluate a model on the test set
def evaluate_model(model, test_images):
    """
    Evaluates the model's performance on a separate test dataset. This function
    outputs the loss and accuracy of the model, providing an estimate of its
    generalization capability.
    """
    results = model.evaluate(test_images, verbose=0)
    print("## Test Loss: {:.5f}".format(results[0]))
    print("## Accuracy on the test set: {:.2f}%".format(results[1] * 100))


# Main script where the image data is loaded, model is built, trained, and evaluated
image_dir = Path(
    'C:\\Users\\HOTZERO\\Desktop\\Diabetic Retina\\Dataset\\gaussian_filtered_images')

image_df = load_image_data(image_dir)
display_images(image_df)

train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True,
                                     random_state=1)
train_gen, test_gen, train_images, val_images, test_images = (
    create_data_generators(image_df, train_df, test_df))

# Modify the training parameters
model = build_model(MobileNetV2, num_classes=5)
history = model.fit(train_images, validation_data=val_images, epochs=10, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                     restore_best_weights=True)])

plot_training_history(history)
evaluate_model(model, test_images)

# Save the model in SavedModel format
model_save_path = "C:\\Users\\HOTZERO\\Desktop\\Diabetic Retina\\Dataset\\my_model.h5"
model.save(model_save_path)


# Additional functions for predictions, classification reports, and confusion matrices
def display_first_predictions(Predictions, num_predictions=5):
    """
    Display the first 'num_predictions' from the predictions list.
    """
    print(f'The first {num_predictions} predictions: {Predictions[:num_predictions]}')


# Generate predictions and create reports
def generate_classification_report(true_labels, predicted_labels, zero_division=1):
    # Generate and print a classification report.

    report = classification_report(true_labels, predicted_labels,
                                   zero_division=zero_division)
    print("Classification Report:\n", report)


def plot_confusion_matrix(y_true, y_pred, ClassName):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    # annot=True to annotate cells with numbers
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(ClassName)
    ax.yaxis.set_ticklabels(ClassName)
    plt.show()


def display_images_with_predictions(df, Predictions, NumRows=3, NumCols=4):
    # Display images with their true labels and predictions.

    fig, axes = plt.subplots(nrows=NumRows, ncols=NumCols, figsize=(10, 7),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        img_path = df.Filepath.iloc[i]
        if os.path.exists(img_path):
            ax.imshow(plt.imread(img_path))
            ax.set_title(f"True: {df.Label.iloc[i]}\nPredicted: {Predictions[i]}")
        else:
            ax.set_title("Image not found.")
    plt.tight_layout()
    plt.show()


"""
 After the model is trained, we use it to predict the class labels for the test dataset. 
 The 'predict' function outputs the probabilities that the input images belong to each class.
"""
predictions = model.predict(test_images)

predictions_indices = np.argmax(predictions, axis=1)

true_classes = test_images.classes

class_names = list(test_images.class_indices.keys())

label_map = train_images.class_indices
label_map = dict((v, k) for k, v in label_map.items())
predictions_labels = [label_map[idx] for idx in predictions_indices]

display_first_predictions(predictions_labels)

generate_classification_report(test_df.Label, predictions_labels)

display_images_with_predictions(test_df, predictions_labels)



