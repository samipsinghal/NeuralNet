# Neural Network from Scratch

This repository contains the implementation of a neural network from scratch based on the tutorial provided by [Adventures in Machine Learning](http://adventuresinmachinelearning.com/neural-networks-tutorial/). The goal of this project is to achieve accurate predictions with limited labeled data.

## Overview

This project demonstrates how to build and train a neural network from scratch using the MNIST dataset. The implementation focuses on understanding the fundamental concepts of neural networks, including feedforward, backpropagation, and gradient descent. The neural network is trained on the MNIST dataset to recognize handwritten digits.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Details](#details)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the notebook and reproduce the results, you need to have Python and the necessary libraries installed. Follow the steps below to set up your environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/samipsinghal/NeuralNetworkFromScratch.git
    cd NeuralNetworkFromScratch
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use this repository, open the `Neural Network from Scratch.ipynb` notebook in Jupyter Notebook or Jupyter Lab. You can run the cells in the notebook to understand the steps involved in the implementation of the neural network and see the results of various experiments.

## Project Structure

The repository is structured as follows:

- `Neural Network from Scratch.ipynb`: Main Jupyter Notebook containing the project implementation.
- `requirements.txt`: List of Python libraries required to run the notebook.

## Details

### Data Preparation

We use the lower resolution MNIST dataset from `sklearn.datasets`. The data is scaled to have a mean of 0 and unit variance to help the algorithm converge.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
