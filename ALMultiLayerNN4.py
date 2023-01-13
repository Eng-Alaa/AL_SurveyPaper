# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:15:11 2022

@author: engal
"""
import argparse
import pandas as pd
#import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Plots import PlotConfMatrix, Plot_PieChart
import torch.nn.functional as F
import math
# Hyper-parameters 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--input_size', type=int, default=784 , help='Input size')
parser.add_argument('--hidden_size', type=int, default=500)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--bacth_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--per_init_labeled', type=float, default=0.05, help="number of init labeled samples")
parser.add_argument('--query_budget', type=int, default=300, help="query budget")
parser.add_argument('--query_size', type=int, default=50, help="query size (iteratively)")
args = parser.parse_args()
np.random.seed(args.seed)

Sampling_method=3 # Random:0 --- Least confident:1 --- Least Margin=2 --- Entropy=3
# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
# split data into initial labeled data and pool
training_data, training_targets= train_dataset.data, train_dataset.targets
n_samples=training_data.shape[0] # no. of training data

# Data shuffling step
training_indices= np.random.choice(n_samples, math.ceil(args.per_init_labeled*n_samples), replace=False)

# select the initial labeled data (size=args.per_init_labeled*n_samples)
X_train = training_data[training_indices]
y_train = training_targets[training_indices]
mask = torch.ones(training_data.shape[0], dtype=torch.bool)
mask[training_indices] = False # mark all selected data (i.e., labeled data)
n_classes= pd.Series(y_train).value_counts()
Plot_PieChart(y_train, 'PieChart', 0) # here, you can plot a pie chart to show the size of each class

# the rest of the data are unlabeled
X_pool = training_data[mask]
y_pool = training_targets[mask]

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_data, test_targets= test_dataset.data, test_dataset.targets
UniqueTestClasses=np.unique(np.array(test_targets))
print("Number of test classes is " + str(len(UniqueTestClasses)))

# show some test images
for i in range(6):
    plt.subplot(2,3,i+1)
    Temp=test_data[i]
    plt.imshow(Temp, cmap='gray')
plt.show()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(args.input_size, args.hidden_size, args.num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  
Query_rounds=math.ceil(args.query_budget/args.query_size)
acc=np.zeros(Query_rounds+1)
# Train the model
def Training(X_train,y_train):
    X_train=X_train.reshape(-1,28*28).to(device, dtype=torch.float32)
    y_train=y_train.to(device)
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, X_train, y_train
      
def softmax(x):
    x=np.float64(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(actual, predicted):
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])    

# train the model on X_train
for epoch in range(args.epochs):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        loss, X_train, y_train= Training(X_train,y_train)
        
        if (epoch+1) % 2 == 0:
            print (f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    test_data=test_data.reshape(-1,28*28).to(device, dtype=torch.float32)
    outputs = model(test_data)
    value, predicted = torch.max(outputs.data, 1)
    conf_matrix=confusion_matrix(test_targets, predicted)
    PlotConfMatrix(conf_matrix, 'Confusion_Matrix', 0)
    n_correct = (predicted == test_targets).sum().item()
    acc[0] = 100.0 * n_correct / test_targets.shape[0]
    print(f'Accuracy of the testing data on the {X_train.shape[0]:} training images: {acc[0]} %')

def Get_data(X_pool, y_pool, Query_Size, X_train, y_train, training_indices):
    New_train_data=X_pool[training_indices]
    New_train_data=New_train_data.reshape(-1,28*28).to(device, dtype=torch.float32)
    X_train=torch.cat((X_train,New_train_data),0)    
    New_train_targets=y_pool[training_indices]
    Plot_PieChart(New_train_targets, 'PieChart_R', q+1)
    y_train=torch.cat((y_train,New_train_targets.to(device)),0)
    mask = torch.ones(X_pool.shape[0], dtype=torch.bool)
    mask[training_indices] = False
    X_pool = X_pool[mask]
    y_pool = np.delete(y_pool, training_indices, axis=0)
    return X_pool, y_pool, X_train, y_train 

def RandomSampling(X_pool, y_pool, Query_Size, X_train, y_train):
    new_samples=X_pool.shape[0]
    RandPer=torch.randperm(new_samples)
    training_indices = RandPer[0:Query_Size]
    return Get_data(X_pool, y_pool, Query_Size, X_train, y_train, training_indices)

# Least margin sampling method
def LeastMargin(X_pool, y_pool, Query_Size, X_train, y_train):
    X_pool=X_pool.reshape(-1,28*28).to(device, dtype=torch.float32)
    outputs = model(X_pool)
    #value, predicted = torch.max(outputs.data, 1)
    prob = F.softmax(outputs,dim=1)
    probs_sorted, idxs = prob.sort(descending=True)
    uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
    training_indices=uncertainties.sort()[1][:Query_Size] # [1] return only inicies
    return Get_data(X_pool, y_pool, Query_Size, X_train, y_train, training_indices)

# least confident sampling method
def LeastConfident(X_pool, y_pool, Query_Size, X_train, y_train):
    X_pool=X_pool.reshape(-1,28*28).to(device, dtype=torch.float32)
    outputs = model(X_pool)
    #value, predicted = torch.max(outputs.data, 1)
    prob = F.softmax(outputs,dim=1)    
    uncertainties = prob.max(1)[0]
    training_indices=uncertainties.sort()[1][:Query_Size]
    return Get_data(X_pool, y_pool, Query_Size, X_train, y_train, training_indices)

# Entropy sampling method
def EntropySampling(X_pool, y_pool, Query_Size, X_train, y_train):
    X_pool=X_pool.reshape(-1,28*28).to(device, dtype=torch.float32)
    outputs = model(X_pool)
    probs = F.softmax(outputs,dim=1)    
    log_probs = torch.log(probs)
    uncertainties = (probs*log_probs).sum(1)  
    training_indices=uncertainties.sort()[1][:Query_Size]
    return Get_data(X_pool, y_pool, Query_Size, X_train, y_train, training_indices)

for q in range(Query_rounds): 
    if Sampling_method==0:
        X_pool, y_pool, X_train, y_train= RandomSampling(X_pool,y_pool, args.query_size, X_train, y_train)
    elif Sampling_method==1:
        X_pool, y_pool, X_train, y_train= LeastConfident(X_pool,y_pool, args.query_size, X_train, y_train)
    elif Sampling_method==2:
        X_pool, y_pool, X_train, y_train= LeastMargin(X_pool,y_pool, args.query_size, X_train, y_train)
    else:
        X_pool, y_pool, X_train, y_train= EntropySampling(X_pool, y_pool, args.query_size, X_train, y_train)
    
    #n_total_steps = len(train_loader)
    for epoch in range(args.epochs):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]            
            loss, X_train, y_train= Training(X_train,y_train)
            if (epoch+1) % 2 == 0:
                print (f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        test_data=test_data.reshape(-1,28*28).to(device, dtype=torch.float32)
        outputs = model(test_data)
        value, predicted = torch.max(outputs.data, 1)
        conf_matrix=confusion_matrix(test_targets, predicted)
        PlotConfMatrix(conf_matrix, 'Confusion_Matrix', q+1) 
        n_correct = (predicted == test_targets).sum().item()
        acc[q+1] = 100.0 * n_correct / test_targets.shape[0]
        print(f'Accuracy of the testing data on the {X_train.shape[0]:} training images: {acc[q+1]} %')

Labels=[]
for k in range(q+2):
    print(3000+k*50)
    #Labels=(Labels + str(3000+q*50))
    Labels.append(str(3000+k*50))
plt.plot(Labels, acc)
plt.ylabel('Accuracy (%)')
plt.xlabel('No. of labeled points')
# plt.savefig('RandomSampling.png')
# plt.savefig('RandomSampling.svg')
plt.show()

    
