#  MNIST Handwritten Digit Classification with PyTorch ✍️

## 📝 Description

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model achieves approximately **98.7% accuracy** on the test set. 🚀 It features custom data loaders for training and testing with transformations like grayscale conversion, resizing, normalization, and batching.

## ✨ Features

-   **🧠 CNN Architecture**: The model consists of two convolutional layers, dropout for regularization, and two fully connected layers.
-   **🏋️‍♂️ Training & Evaluation**: Includes complete loops for training and evaluating the model, tracking both loss and accuracy.
-   **⚡ GPU Acceleration**: Automatically utilizes a CUDA-enabled GPU for training if one is available, otherwise, it defaults to the CPU.
-   **🖼️ Data Visualization**: Provides a script to visualize random samples from the test set along with their predicted and actual labels.

## 📂 Dataset

The model expects the MNIST dataset to be organized in a specific folder structure, compatible with `torchvision.datasets.ImageFolder`. You should have a main `Dataset` directory with `train` and `test` subdirectories. Inside each of these, there should be folders named 0 through 9, containing the respective digit images.


Dataset/

├── train/

│   ├── 0/

│   ├── 1/

│   ├── ...

│   └── 9/

└── test/

├── 0/

├── 1/

├── ...

└── 9/


**❗️ Note**: You will need to update the dataset path in the Python script to point to your local `Dataset` directory.

## 🛠️ Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install torch torchvision matplotlib
    ```

## 🚀 Usage

Here’s how to run the different parts of the project.

1.  **Train the Model:**
    To train the model for 10 epochs, execute the main script. The training progress and loss will be printed to the console.
    ```bash
    python your_script_name.py
    ```

2.  **Evaluate the Model:**
    The evaluation on the test set is performed after each training epoch. The final accuracy is displayed after the last epoch.

3.  **Visualize Predictions:**
    The script includes a section to pick a random test image, display it, and print the model's prediction versus the actual label. This runs automatically after the training and evaluation loop is complete.

## 📊 Results

The model's performance improves with each epoch as the training loss decreases. After 10 epochs, the model achieves a high accuracy on the test set.

**Sample Training Log:**


Train Epoch: 1 [58000/60000( 97%)]	0.303106
Test set: Average loss: 0.1095, Accuracy 9666/10000 (96.66%)
...
Train Epoch: 10 [58000/60000( 97%)]	0.166659
Test set: Average loss: 0.0384, Accuracy 9874/10000 (98.74%)


**Final Test Accuracy:** 🎉 **~98.7%**

## 🔮 Future Improvements

-   **🔧 Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizer settings to potentially improve accuracy.
-   **💾 Model Saving and Loading**: Implement functionality to save the trained model weights and load them for inference without retraining.
-   **🔄 Data Augmentation**: Introduce more advanced data augmentation techniques (e.g., rotation, shifting) to make the model more robust.
-   **🏗️ Different Architectures**: Explore other CNN architectures, such as ResNet or VGG, to compare performance.


