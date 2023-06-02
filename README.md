# TSAI_Assignment5
This is assignment 5 solution for TSAI 

## This repository contains 3 files : 
1. utils.py  
2. model.py 
3. s5.ipynb  

## 1. utils.py  
This file is where the train and test functions are defined . 
The train function is a training loop for a neural network model.  This function performs training process by iterating over batches of training data , making predictions , calculating loss , backpropogating gradients and finally updating model parameters. It also keeps track of training accuracy and loss values during the training process. 
The test function is used to evaluate a trained model on a test dataset.  It iterates over the test dataset makes predictions and accumulates test loss. Later it calculates the average loss and accuracy 

## 2. model.py  
This script defines the CNN model with 4 convolution layers and 2 fully connected layers.The forward method specifies the forward pass computation. 
In the constructor method , layers of neural network are defined.  
    a. first convolutional layer with 1 input channel, 32 output channels, and a kernel size of 3x3.
    b. second convolutional layer with 32 input channels, 64 output channels, and a kernel size of 3x3.
    c. third convolutional layer with 64 input channels, 128 output channels, and a kernel size of 3x3.
    d. fourth convolutional layer with 128 input channels, 256 output channels, and a kernel size of 3x3.
    e. first fully connected (linear) layer with 4096 input features and 50 output features.
    f. second fully connected layer with 50 input features and 10 output features.

## 3. s5.ipynb  
This is the main file where the Mnist dataset is downloaded and transformations are applied on train and test dataset. 
This is also where the model object is instantiated and train and test functions are called .Graphs of train and test accuracy and loss are shown as an output. 
