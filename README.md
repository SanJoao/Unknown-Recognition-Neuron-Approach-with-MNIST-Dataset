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
  
---
#### Experiment Setup

1. **Data Preparation**
   - **MNIST Dataset**: Used as the base dataset for training. Contains images of handwritten digits (0-9).
   - **Additional Images**: Added images from the Quick, Draw! dataset, the National Museum of American History Dataset, and generated random noise images. These were labeled as 'unknown'.

   ```python
   from tensorflow.keras.datasets import mnist
   from PIL import Image
   import numpy as np
   import os

   # Load MNIST data
   (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
   train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
   test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
   train_images, test_images = train_images / 255.0, test_images / 255.0

   def load_non_digit_images(folder_path):
       images = []
       labels = []
       for filename in os.listdir(folder_path):
           if filename.startswith("Unknown"):
               img_path = os.path.join(folder_path, filename)
               img = Image.open(img_path).convert('L')
               img = img.resize((28, 28))
               img_array = np.array(img) / 255.0
               images.append(img_array.reshape(28, 28, 1))
               labels.append(10)  # Label for 'unknown'
       return np.array(images), np.array(labels)

   # Example paths to non-digit image folders
   non_digit_train_path = "/path/to/training/non_digit_images"
   non_digit_train_images, non_digit_train_labels = load_non_digit_images(non_digit_train_path)
   ```

2. **Model Architecture**
   - The model includes convolutional layers followed by max pooling and dense layers, ending with a softmax output layer that has 11 outputs (10 for each digit and 1 for 'unknown').

   ```python
   from tensorflow.keras import layers, models

   model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dense(11, activation='softmax')  # Including 'unknown' class
   ])
   ```

3. **Training the Model**
   - The model is trained using combined datasets, including both MNIST and non-digit images, using sparse categorical crossentropy as the loss function.

   ```python
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   history = model.fit(train_images_combined, train_labels_combined, epochs=10,
                       validation_data=(test_images_combined, test_labels_combined))
   ```

#### Results and Analysis
- **Performance**: The model achieved a high accuracy rate in classifying MNIST digits and effectively recognized 'unknown' images.
- **Confusion Matrix and Classification Report**:
  - A confusion matrix and classification report were generated to analyze the performance across all classes, highlighting the effectiveness of the 'unknown' neuron.

  ```python
  from sklearn.metrics import classification_report, confusion_matrix
  import seaborn as sns
  import matplotlib.pyplot as plt

  pred_probs = model.predict(test_images_combined)
  predictions = np.argmax(pred_probs, axis=1)
  conf_matrix = confusion_matrix(test_labels_combined, predictions)

  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()

  print(classification_report(test_labels_combined, predictions))
  ```

#### Conclusion
This experiment demonstrates a potential method for improving OOD detection in neural networks by incorporating an additional 'unknown' output neuron. Future work will focus on refining the model architecture and exploring other datasets to further validate and enhance this approach.

---
### Project Directory Overview

**1. `model.h5`:** Contains the trained CNN model including architecture, weights, and configuration.

**2. `Combined__Training` and `Combined__Test`:** These folders hold the datasets for training and testing. Each includes:
   - **Noise:** Random noise images.
   - **NMAH:** Grayscale images from the National Museum of American History, resized to 28x28 pixels.
   - **Quick_Draw:** Processed grayscale images from the Quick, Draw! dataset.

**3. `Other__Test`:** Contains processed evaluation datasets, each with 1000 grayscale images (28x28 pixels) from various sources such as Cifar-10, ImageNet, EMNIST, Fashion MNIST, additional MNIST images, MNIST-M, and Omniglot.

**4. MNIST Dataset:** Not included in the folders due to easy availability for import during training and testing phases.

### Additional Notes
- **Image Format:** All images are formatted to grayscale and resized to 28x28 pixels to match the model’s input requirements.
- **Usage:** The `model.h5` file can be directly loaded with TensorFlow/Keras for further evaluation or modification.

