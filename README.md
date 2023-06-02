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
This script defines the neural network model 

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)-   ####Defines the first convolutional layer with 1 input channel, 32 output channels, and a kernel size of 3x3.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



## 3. s5.ipynb  
This is the main file where the Mnist dataset is downloaded and transformations applied on train and test dataset. 
