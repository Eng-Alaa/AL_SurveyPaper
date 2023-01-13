# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:51:08 2022

@author: engal
"""
import matplotlib.pyplot as plt
import numpy as np

def PlotConfMatrix(conf_matrix, FileName, q):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    #FileNameNew=(FileName + '.png')
    plt.savefig(FileName + str(q)+  '.png')
    #plt.savefig('ConfusionMatrix.png')
    plt.savefig(FileName + str(q) + '.eps')
    plt.savefig(FileName + str(q) + '.pdf')
    plt.savefig(FileName + str(q) + '.svg')
    plt.show()

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:d}\n({:.1f}%)".format(absolute, pct)

def Plot_PieChart(y_train, FileName, q):
    #FileName='PieChart'
    labels = '0', '1', '2', '3', '4', '5', '6', '7','8', '9'
    frequency, bins = np.histogram(y_train, bins=10, range=[0, 10])
    fig1, ax1 = plt.subplots()
    #ax1.pie(frequency, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    #ax1.pie(frequency, labels=labels, autopct=lambda pct: func(pct, frequency), shadow=True, startangle=90)
    ax1.pie(frequency, labels=labels, autopct=lambda p : '{:.2f}% \n ({:,.0f})'.format(p,p * sum(frequency)/100), shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(FileName + str(q)+  '.png')
    #plt.savefig('ConfusionMatrix.png')
    plt.savefig(FileName + str(q) + '.eps')
    plt.savefig(FileName + str(q) + '.pdf')
    plt.savefig(FileName + str(q) + '.svg')
    plt.show()

def Plot_Histogram(y_train):
    frequency, bins = np.histogram(y_train, bins=10, range=[0, 10])
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(y_train, bins=10)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency');

