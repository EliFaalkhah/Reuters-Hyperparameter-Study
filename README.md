# Skin Cancer Classification: Malignant vs. Benign

## 1. Project Overview
This project implements a Convolutional Neural Network (CNN) to classify skin lesion images as either malignant or benign. The model is built using the TensorFlow and Keras frameworks and leverages transfer learning techniques to achieve robust performance. The entire workflow, from data preprocessing and augmentation to model training and evaluation, is documented in the accompanying Jupyter Notebook.

## 2. Dataset
This project utilizes the "Skin Cancer: Malignant vs. Benign" dataset from Kaggle. The dataset is composed of thousands of JPEG images of skin lesions, pre-categorized into two classes.

The data was split into training and testing sets with the following distribution:

**Training Set:**
- Benign Images: 1,440
- Malignant Images: 1,197

**Testing Set:**
- Benign Images: 360
- Malignant Images: 300

## 3. Model Architecture
The model is a Sequential CNN that uses a pre-trained model as its convolutional base for feature extraction, followed by a custom-built classifier.

The architecture of the custom classifier is as follows:
- **Flatten layer:** Converts the feature maps into a single vector.
- **Dense layer (1):** 512 units with a ReLU activation function.
- **Dropout layer (1):** Rate of 0.5 to prevent overfitting.
- **Dense layer (2):** 256 units with a ReLU activation function.
- **Dropout layer (2):** Rate of 0.5.
- **Dense output layer:** 2 units with a softmax activation function for multi-class classification.

**Model Parameters:**
- Total Parameters: 33,001,794
- Trainable Parameters: 12,977,410
- Non-trainable Parameters: 20,024,384 (frozen layers of the pre-trained base model)

## 4. Training Process
The model was compiled and trained with the following configuration:
- **Optimizer:** Adam
- **Loss Function:** `categorical_crossentropy`
- **Metrics:** Accuracy
- **Epochs:** 12
- **Data Augmentation:** `ImageDataGenerator` used to rescale pixel values from $[0, 255]$ to $[0, 1]$.
- **Callbacks:** `ReduceLROnPlateau` employed to monitor validation accuracy and reduce the learning rate if it stops improving.

## 5. Results
The model's performance was evaluated on the unseen test set after the training was complete.
- **Final Validation Accuracy (at epoch 12):** ~81.7%
- **Final Test Loss:** 0.4345
- **Final Test Accuracy:** 79.69%

This indicates that the model can correctly classify a previously unseen skin lesion image with approximately 80% accuracy.
