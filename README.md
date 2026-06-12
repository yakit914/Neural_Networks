# Neural Network from Scratch

Building a neural network from scratch (no PyTorch, no TensorFlow, just NumPy) to understand how it actually works
<br>
(e.g., forward passes, backpropagation, and optimization)

## What I Implemented

- **Dense layers** - Matrix math for weights and biases
- **Activation functions** - ReLU, LeakyReLU, Softmax
- **Loss functions** - Categorical Crossentropy for classification
- **Backpropagation** - Computing gradients through the network
- **Optimizers** - SGD with learning rate decay
- **Training loop** - Mini-batch processing, validation, epoch tracking

## The Use Case - MNIST

Using MNIST (handwritten digits) as a concrete example to test the network. 60,000 training images, 10,000 test images (28×28 pixels each). This is just the test case - the concepts apply to any classification problem.

**Original source:** [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)  
**Data from:** [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

**Credits to:** Yann LeCun (NYU), Corinna Cortes (Google Labs), Christopher J.C. Burges (Microsoft Research)

## Project Structure

```
Neural_Networks/
├── Neural_Network_mod/       # Core neural network code
│   ├── Activation.py         # Activation functions
│   ├── Layer.py              # Dense layer
│   ├── Loss.py               # Loss functions
│   ├── Neural_Network.py     # Main network class
│   └── Optimizer.py          # SGD optimizer
├── Use_Case/
│   └── MNIST_Dataset.ipynb   # Training & testing notebook
└── input/                    # MNIST data files
```

## Getting Started

### Setup

1. Download MNIST data from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and put it in the `input/` folder
2. Install dependencies:

   ```
   pip install numpy matplotlib jupyter
   ```

   - **numpy** - All the matrix math
   - **matplotlib** - Plotting results and visualizing confusion matrix
   - **jupyter** - Running the notebook

3. Run the notebook:
   ```
   cd Use_Case
   jupyter notebook MNIST_Dataset.ipynb
   ```

### What the notebook does

- Loads and normalizes the MNIST images
- Sets up the neural network (784 → 128 → 128 → 64 → 10)
- Trains for 25 epochs with SGD + learning rate decay
- Tests on 10,000 test images
- Shows loss curves, accuracy, confusion matrix

## Results

Gets around **97-98% accuracy** on the test set. Good enough to verify everything works!

The notebook outputs:

- Training/validation loss and accuracy curves
- Test set accuracy and loss
- Confusion matrix (which digits get confused with which)
- Examples of misclassified digits

## Key Learning Takeaways

- **Forward pass** - How data flows through layers
- **Backpropagation** - Computing gradients through the chain rule
- **Gradient descent** - Using gradients to update weights
- **Mini-batching** - Why we train on batches instead of all data at once
- **Validation** - How to detect overfitting
- **Optimizers** - Why learning rate decay helps convergence
