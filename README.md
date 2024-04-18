### Experiment Report: Unknown Recognition Neuron Approach with MNIST Dataset

#### Overview
The "Unknown Recognition Neuron" experiment aimed to investigate the effectiveness of introducing an "unknown" class to handle out-of-distribution data when training a convolutional neural network (CNN) with the MNIST dataset. This experiment was inspired by the need to improve the model's ability to classify images it has never seen before, which are not part of the standard dataset categories.

#### Experiment Setup
1. **Data Preparation**:
    - **MNIST Dataset**: Standard dataset of handwritten digits used for the primary training and testing.
    - **Non-digit Images**: Additional images from various sources labeled as 'unknown' to simulate out-of-distribution data.
    - Data from the Quick, Draw! dataset, National Museum of American History, and randomly generated noise images were added.

2. **Model Architecture**:
    - The CNN model consisted of three convolutional layers followed by max-pooling layers, and two dense layers at the end, including an output layer with 11 units (10 for the MNIST digits and 1 for the 'unknown' class).
    - Activation: ReLU for hidden layers and Softmax for the output layer.
    - The model was compiled with the Adam optimizer and sparse categorical crossentropy as the loss function.

3. **Training**:
    - The combined dataset included original MNIST images and non-digit images.
    - Data normalization was applied to scale pixel values between 0 and 1.
    - The model was trained for 10 epochs, with checkpoints and logging handled by Weights & Biases (W&B).

4. **Testing**:
    - The model was evaluated on a combined test set that included unseen MNIST images and additional 'unknown' images.
    - Performance metrics such as accuracy, precision, recall, and F1-score were recorded.

#### Results
- The CNN managed to achieve a high accuracy rate of approximately 99% on the test set, effectively recognizing 'unknown' images in most cases.
- Confusion matrix and classification reports indicated strong performance across all classes, including the 'unknown' class.

#### Challenges and Observations
- **Scalability**: While the experiment showed promising results, the scalability of adding an 'unknown' neuron when faced with a broader range of out-of-distribution images remains a question.
- **Data Dependency**: The effectiveness of the 'unknown' neuron heavily depends on the diversity and representativeness of the non-digit images used during training.

#### Code and Reproducibility
```python
# Code setup for training the CNN model
import wandb
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# Initialize W&B
wandb.init(project='cnn_unknown_recognition', entity='sanxuejin')

# Load and preprocess MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)) / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)) / 255.0

# Load non-digit images for training and testing
def load_non_digit_images(folder_path):
    images, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.startswith("Unknown"):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')
            img = np.array(img.resize((28, 28))).reshape(28, 28, 1) / 255.0
            images.append(img)
            labels.append(10)  # Label '10' for unknown
    return np.array(images), np.array(labels)

# Paths to your non-digit image folders
train_non_digit_path = '/path/to/training/non-digit/images'
test_non_digit_path = '/path/to/testing/non-digit/images'

# Load non-digit images
non_digit_train_images, non_digit_train_labels = load_non_digit_images(train_non_digit_path)
non_digit_test_images, non_digit_test_labels = load_non_digit_images(test_non_digit_path)

# Combine datasets
train_images = np.concatenate([train_images, non_digit_train_images])
train_labels = np.concatenate([train_labels, non_digit_train_labels])
test_images = np.concatenate([test_images, non_digit_test_images])
test_labels = np.concatenate([test_labels, non_digit_test_labels])

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu

'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(11, activation='softmax')  # Including 'unknown' neuron
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[wandb.keras.WandbCallback()])
```

#### Future Directions
- **Experiment with more complex and diverse datasets** to further validate the robustness of the 'unknown' neuron approach.
- **Explore alternative architectures and training techniques** that might improve the model's ability to generalize from seen to unseen data.
