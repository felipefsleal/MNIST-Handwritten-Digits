# MNIST Handwritten Digit Classification with PyTorch

This project is a Convolutional Neural Network (CNN) built using PyTorch for classifying handwritten digits from the famous MNIST dataset. The model is trained and evaluated in a Jupyter Notebook.

---

## Dataset 📖

The project uses the **MNIST (Modified National Institute of Standards and Technology)** dataset, which is a classic benchmark in the machine learning community.

* It consists of **70,000 grayscale images** of handwritten digits (0-9).
* Each image is **28x28 pixels**.
* The dataset is pre-divided into 60,000 training images and 10,000 testing images.

The repository's `Dataset/` folder is structured to be compatible with PyTorch's `ImageFolder` class:

```
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
```

---

## Model Architecture 🧠

The CNN architecture is defined as follows:

1.  **Convolutional Layer 1 (`Conv2d`)**:
    * Input channels: 1 (grayscale)
    * Output channels: 10
    * Kernel size: 5x5
2.  **Max Pooling Layer (`MaxPool2d`)**:
    * Kernel size: 2x2
    * Activation: ReLU
3.  **Convolutional Layer 2 (`Conv2d`)**:
    * Input channels: 10
    * Output channels: 20
    * Kernel size: 5x5
4.  **Dropout Layer (`Dropout2d`)**: Randomly zeroes some of the elements of the input tensor.
5.  **Max Pooling Layer (`MaxPool2d`)**:
    * Kernel size: 2x2
    * Activation: ReLU
6.  **Fully Connected Layer 1 (`Linear`)**:
    * Input features: 320
    * Output features: 50
    * Activation: ReLU
7.  **Dropout Layer (`Dropout`)**
8.  **Fully Connected Layer 2 (`Linear`)**:
    * Input features: 50
    * Output features: 10 (one for each digit class)

### Training Details

* **Optimizer**: Adam (`lr=0.001`)
* **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`)
* **Epochs**: 10
* **Batch Size**: 100

---

## Installation & Requirements ⚙️

To run this project, you'll need Python 3 and the following libraries.

### 1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Set up a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install the required packages:

The main dependencies are:
* `torch`
* `torchvision`
* `matplotlib`
* `jupyter`

You can install them all using pip:

```bash
pip install torch torchvision matplotlib jupyter
```

---

## How to Run 🚀

Follow these steps to execute the project:

1.  Ensure you have completed the installation steps above.
2.  Navigate to the `Project/` directory.
3.  Launch the Jupyter Notebook server:

    ```bash
    jupyter notebook
    ```

4.  Open the `.ipynb` notebook file from the Jupyter interface in your browser.
5.  Run all the cells in the notebook to train the model and see the results.

---

## Results ✨

After training for 10 epochs, the model is evaluated on the test dataset. The key metrics are:

* **Test Loss**: The average loss calculated on the 10,000 test images.
* **Test Accuracy**: The percentage of test images correctly classified by the model (typically achieves **>98% accuracy**).

The notebook also includes a section to visualize a **random prediction**, where it selects a random image from the test set, displays it, and prints the model's prediction versus the actual label.



---

## Future Improvements 💡

This project serves as a solid baseline. Here are some ways it could be extended:

* **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizer settings to improve performance.
* **Advanced Architectures**: Implement more complex CNN architectures like VGG, ResNet, or DenseNet.
* **Data Augmentation**: Apply transformations (e.g., rotation, scaling) to the training data to make the model more robust.
* **Regularization**: Add techniques like Batch Normalization to prevent overfitting and stabilize training.

---
