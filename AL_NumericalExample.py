# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:57:15 2022

@author: engal
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import sklearn
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.special import entr
from sklearn.metrics import accuracy_score
import operator
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split

#Temp=np.transpose(np.array([[1, 2, 2,3,3,4,5,5,5,6], [1, 4, 3,3,5,2,1,3,5,4]]))
#df = pd.DataFrame(Temp, columns=['a', 'b'])
#Labels=np.array([1,1,1,1,1,2,2,2,2,2])


# Prepare training and testing
#TrainingData = pd.DataFrame(np.transpose(np.array([[2, 2,4,5], [2, 4,3,4]])))
TrainingData = pd.DataFrame(np.transpose(np.array([[1,2,6], [1,4,4]])))
#TrainingLabels=np.array([1,1,2,2])
TrainingLabels=np.array([1,2,3])

TrainingData=TrainingData.to_numpy()
# Plot our training data
#plt.figure(figsize=(8.5, 6), dpi=130)
#plt.plot(np.transpose(TrainingData[TrainingLabels==1,0]),np.transpose(TrainingData[TrainingLabels==1,1]), 'ro')
#plt.plot(np.transpose(TrainingData[TrainingLabels==2,0]),np.transpose(TrainingData[TrainingLabels==2,1]), 'go')
#plt.plot(np.transpose(TrainingData[TrainingLabels==3,0]),np.transpose(TrainingData[TrainingLabels==3,1]), 'bo')
#plt.show()

# the pool of unlabeled data points
TestingData = pd.DataFrame(np.transpose(np.array([[2,1,2,3,3,2,1,5,6,4], [1,2,2,2,3,3,4,3,2,3]])))
TestingLabels=np.array([1,1,1,1,2,2,2,3,3,3])
TestingData=TestingData.to_numpy()
# Plot our training/labeled data and unlabeled data
plt.figure(figsize=(8.5, 6), dpi=130)
plt.title("Initial labeled data and unlabeled data")
plt.plot(np.transpose(TrainingData[TrainingLabels==1,0]),np.transpose(TrainingData[TrainingLabels==1,1]), 'ro', markersize=20)
plt.plot(np.transpose(TrainingData[TrainingLabels==2,0]),np.transpose(TrainingData[TrainingLabels==2,1]), 'go', markersize=20)
plt.plot(np.transpose(TrainingData[TrainingLabels==3,0]),np.transpose(TrainingData[TrainingLabels==3,1]), 'bo', markersize=20)
plt.plot(np.transpose(TestingData[:,0]),np.transpose(TestingData[:,1]), 'ks', markersize=10)
for i in range(TestingData.shape[0]):
    Text='x_' + str(i+1)
    plt.text(TestingData[i,0]-0.13, TestingData[i,1]-0.13,Text,fontsize=9)
plt.show()

# train a model on TrainingData, calculate the accuracy of TestingData
clf = RandomForestClassifier(n_estimators = 10, random_state = 1) 
clf.fit(TrainingData, TrainingLabels) 
# calculate prediction probabilities of the unlabeled data
predictions = clf.predict(TestingData) 
Accuracy= np.array(accuracy_score(predictions, TestingLabels))
Accuracy=Accuracy.reshape((1,1))
predicted_probs = clf.predict_proba(TestingData) 
print(predicted_probs) 
entropy_array = entr(predicted_probs).sum(axis=1)
print(entropy_array)

plt.figure(figsize=(8.5, 6), dpi=130)
plt.title("Active learner class predictions after one iteration")
plt.plot(np.transpose(TrainingData[TrainingLabels==1,0]),np.transpose(TrainingData[TrainingLabels==1,1]), 'ro', markersize=20)
plt.plot(np.transpose(TrainingData[TrainingLabels==2,0]),np.transpose(TrainingData[TrainingLabels==2,1]), 'go', markersize=20)
plt.plot(np.transpose(TrainingData[TrainingLabels==3,0]),np.transpose(TrainingData[TrainingLabels==3,1]), 'bo', markersize=20)
plt.plot(np.transpose(TestingData[predictions==1,0]),np.transpose(TestingData[predictions==1,1]), 'rs', markersize=10)
plt.plot(np.transpose(TestingData[predictions==2,0]),np.transpose(TestingData[predictions==2,1]), 'gs', markersize=10)
plt.plot(np.transpose(TestingData[predictions==3,0]),np.transpose(TestingData[predictions==3,1]), 'bs', markersize=10)
for i in range(TestingData.shape[0]):
    Text='x_' + str(i+1)
    plt.text(TestingData[i,0]-0.13, TestingData[i,1]-0.13,Text,fontsize=9)
plt.show()

# select the most uncertain point (with max entropy) in the testing data 
index, value = max(enumerate(entropy_array), key=operator.itemgetter(1))

# least confident
#uncertainties = np.amax(predicted_probs, axis=1)
#index=np.where(uncertainties == uncertainties.min())

# least margin
#predicted_probs.sort()
#uncertainties = predicted_probs[:, -1] - predicted_probs[:,-2]
#index=np.where(uncertainties == uncertainties.min())
print('The most uncertain point is No.', index)

# add the new annotated point to the training data
TrainingData=np.concatenate((TrainingData,np.array(TestingData[index,:]).reshape(1,2)),axis=0)
TrainingLabels= np.append(TrainingLabels,np.array(TestingLabels[index]).reshape(1,1))
# delete the annotated point from the testing data (pool)
TestingData=np.delete(TestingData,index, axis=0)
TestingLabels=np.delete(TestingLabels,index, axis=0)

# Iteratively annotate one point
for i in range(3):
    clf.fit(TrainingData, TrainingLabels) 
    predictions = clf.predict(TestingData) 
    Accuracy= np.append(Accuracy, np.array(accuracy_score(predictions, TestingLabels)))
    plt.figure(figsize=(8.5, 6), dpi=130)
    plt.title("Active learner class predictions after " + str(i+2) +  " iterations (Accuracy: {Acc:.3f})".format(Acc=np.array(accuracy_score(predictions, TestingLabels))))
    plt.plot(np.transpose(TrainingData[TrainingLabels==1,0]),np.transpose(TrainingData[TrainingLabels==1,1]), 'ro', markersize=20)
    plt.plot(np.transpose(TrainingData[TrainingLabels==2,0]),np.transpose(TrainingData[TrainingLabels==2,1]), 'go', markersize=20)
    plt.plot(np.transpose(TrainingData[TrainingLabels==3,0]),np.transpose(TrainingData[TrainingLabels==3,1]), 'bo', markersize=20)
    plt.plot(np.transpose(TestingData[predictions==1,0]),np.transpose(TestingData[predictions==1,1]), 'rs', markersize=10)
    plt.plot(np.transpose(TestingData[predictions==2,0]),np.transpose(TestingData[predictions==2,1]), 'gs', markersize=10)
    plt.plot(np.transpose(TestingData[predictions==3,0]),np.transpose(TestingData[predictions==3,1]), 'bs', markersize=10)
    for i in range(TestingData.shape[0]):
        Text='x_' + str(i+1)
        plt.text(TestingData[i,0]-0.13, TestingData[i,1]-0.13,Text,fontsize=9)
    plt.show()
    
    predicted_probs = clf.predict_proba(TestingData) 
    print(predicted_probs) 
    entropy_array = entr(predicted_probs).sum(axis=1)
    print(entropy_array)
    index, value = max(enumerate(entropy_array), key=operator.itemgetter(1))
    print('The most uncertain point is No.', index)
    
    # add the new selected point to the labeled data
    TrainingData=np.concatenate((TrainingData,np.array(TestingData[index,:]).reshape(1,2)),axis=0)
    TrainingLabels= np.append(TrainingLabels,np.array(TestingLabels[index]).reshape(1,1))
    
    # delete the annotated point from the testing data (pool)
    TestingData=np.delete(TestingData,index, axis=0)
    TestingLabels=np.delete(TestingLabels,index, axis=0)

#print(Accuracy)







