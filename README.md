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

