# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:23:00 2022

@author: engal
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 22:16:27 2022

@author: engal
"""

from dataclasses import replace
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.special import entr
from sklearn.metrics import accuracy_score
import operator
from sklearn.metrics import confusion_matrix
from Plots import PlotConfMatrix
#from sklearn.neighbors import KNeighborsClassifier
#from modAL.models import ActiveLearner
# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

iris = load_iris()
X_raw = iris['data']
y_raw = iris['target']

pca = PCA(n_components=2)
transformed_iris = pca.fit_transform(X=X_raw)

# Isolate the data we'll need for plotting.
x_component, y_component = transformed_iris[:, 0], transformed_iris[:, 1]

# Plot our dimensionality-reduced (via PCA) dataset.
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x_component[y_raw==0], y_component[y_raw==0], color='red',s=50, alpha=8/10, label='Class 1')
plt.scatter(x_component[y_raw==1], y_component[y_raw==1], color='blue',s=50, alpha=8/10, label='Class 2')
plt.scatter(x_component[y_raw==2], y_component[y_raw==2], color='green',s=50, alpha=8/10, label='Class 3')
#plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
#plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
plt.gca().legend(loc="lower right")
#plt.legend(loc="upper left")
plt.title('Iris classes after PCA transformation')
plt.savefig('OriginalIrisData.svg')
plt.savefig('OriginalIrisData.pdf')
plt.show()

# plot data without labels (unlabeled data)
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x_component, y_component, color='black',s=50, alpha=8/10, label='Class 1')
#plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
#plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
#plt.gca().legend(loc="lower right")
plt.title('Unlabeled data')
plt.savefig('OriginalIrisUnData.svg')
plt.savefig('OriginalIrisUnData.pdf')
plt.show()
n_labeled_examples = X_raw.shape[0]

# pick three points randomly and make their labels available (initial training data)
training_indices=np.random.permutation(n_labeled_examples-1)[0:3]
print(training_indices)

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x_component, y_component, color='black',s=50, alpha=8/10, label='Class 1')
#plt.scatter(x=x_component[training_indices], y=y_component[training_indices], color='yellow', cmap='viridis', marker='o')
for i in range(3):
    if y_train[i]==0:
        plt.scatter(x=x_component[training_indices[i]], y=y_component[training_indices[i]], color='red', cmap='viridis', marker='o')
    elif y_train[i]==1:
        plt.scatter(x=x_component[training_indices[i]], y=y_component[training_indices[i]], color='blue', cmap='viridis', marker='o')
    else:
        plt.scatter(x=x_component[training_indices[i]], y=y_component[training_indices[i]], color='green', cmap='viridis', marker='o')        
plt.title('First selected training/labeled points')
plt.savefig('FirstSelectedData.svg')
plt.savefig('FirstSelectedData.pdf')
plt.show()

clf = RandomForestClassifier(max_depth = 4, min_samples_split=2, n_estimators = 200, random_state = 1) 
clf.fit(X_train, y_train) 
predictions = clf.predict(X_raw) 
#predictions = clf.predict(X_pool) 
unqueried_score= np.array(accuracy_score(predictions, y_raw))
#unqueried_score= np.array(accuracy_score(predictions, y_pool))
predicted_probs = clf.predict_proba(X_pool) 
print(predicted_probs) 
entropy_array = entr(predicted_probs).sum(axis=1)
query_index, value = max(enumerate(entropy_array), key=operator.itemgetter(1))
print('The most uncertain point is No.', query_index)
conf_matrix=confusion_matrix(y_raw, predictions)
PlotConfMatrix(conf_matrix, 'Confusion_Matrix', 0) 

is_correct = (predictions == y_raw)
#is_correct = (predictions == y_pool)
# Plot our classification results.
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x_component, y_component, color='gray',s=50, alpha=8/10, label='Unlabeled point')
plt.scatter(x=x_component[is_correct],  y=y_component[is_correct],  color='gold', marker='+', label='Correct',   alpha=8/10)
plt.scatter(x=x_component[~is_correct], y=y_component[~is_correct], color='black', marker='x', label='Incorrect', alpha=8/10)
plt.scatter(x=x_component[training_indices[y_raw[training_indices]==0]], y=y_component[training_indices[y_raw[training_indices]==0]], c='red', marker='o', label='Labeled point (C1)', alpha=8/10)
plt.scatter(x=x_component[training_indices[y_raw[training_indices]==1]], y=y_component[training_indices[y_raw[training_indices]==1]], c='blue', marker='o', label='Labeled point (C2)', alpha=8/10)
plt.scatter(x=x_component[training_indices[y_raw[training_indices]==2]], y=y_component[training_indices[y_raw[training_indices]==2]], c='green', marker='o', label='Labeled point (C3)', alpha=8/10)
plt.title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
plt.legend(loc='lower right')
plt.savefig('FirstSelectedPerformanec.svg')
plt.savefig('FirstSelectedPerformanec.pdf')
plt.show()

N_QUERIES = 10
performance_history = [unqueried_score]

# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).
for index in range(N_QUERIES):
    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    # add the new labeled point to the training data
    training_indices=np.append(training_indices,query_index)
    X_train=np.concatenate((X_train,X),axis=0)
    y_train= np.append(y_train,y)
    # Remove the queried instance from the unlabeled pool.
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
    clf.fit(X_train, y_train) 
    predictions = clf.predict(X_raw) 
    unqueried_score= np.array(accuracy_score(predictions, y_raw))
    predicted_probs = clf.predict_proba(X_pool) 
    print(predicted_probs) 
    entropy_array = entr(predicted_probs).sum(axis=1)
    query_index, value = max(enumerate(entropy_array), key=operator.itemgetter(1))
    print('The most uncertain point is No.', query_index)    # Calculate and report our model's accuracy.
    conf_matrix=confusion_matrix(y_raw, predictions)
    PlotConfMatrix(conf_matrix, 'Confusion_Matrix', index+1) 

    # Save our model's performance for plotting.
    performance_history.append(unqueried_score)    
    is_correct = (predictions == y_raw)
    
    # Plot our updated classification results once we've trained our learner.
    plt.figure(figsize=(8.5, 6), dpi=130)
    plt.scatter(x_component, y_component, color='gray',s=50, alpha=8/10, label='Unlabeled point')
    plt.scatter(x=x_component[is_correct],  y=y_component[is_correct],  color='gold', marker='+', label='Correct',   alpha=8/10)
    plt.scatter(x=x_component[~is_correct], y=y_component[~is_correct], color='black', marker='x', label='Incorrect', alpha=8/10)
    plt.scatter(x=x_component[training_indices[y_raw[training_indices]==0]] , y=y_component[training_indices[y_raw[training_indices]==0]], c='red', marker='o', label='Labeled point (C1)', alpha=8/10)
    plt.scatter(x=x_component[training_indices[y_raw[training_indices]==1]] , y=y_component[training_indices[y_raw[training_indices]==1]], c='blue', marker='o', label='Labeled point (C2)', alpha=8/10)
    plt.scatter(x=x_component[training_indices[y_raw[training_indices]==2]] , y=y_component[training_indices[y_raw[training_indices]==2]], c='green', marker='o', label='Labeled point (C3)', alpha=8/10)
    plt.title('Classification accuracy after {n} queries: {final_acc:.3f}'.format(n=N_QUERIES, final_acc=performance_history[-1]))
    plt.legend(loc='lower right')
    FileName='SelectedPerformance'    
    plt.savefig(FileName + str(index+1)+  '.svg')
    plt.savefig(FileName + str(index+1)+  '.pdf')
    plt.show()
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=unqueried_score))
    #input("Press Enter to continue...")
    
# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')
plt.savefig('FinalAcc.svg')
plt.savefig('FinalAcc.pdf')
plt.show()


#def ActiveLearner(X_train,y_train, X_raw,y_raw,X_pool):
#    clf.fit(X_train, y_train) 
#    predictions = clf.predict(X_raw) 
#    unqueried_score= np.array(accuracy_score(predictions, y_raw))
#    predicted_probs = clf.predict_proba(X_pool) 
#    print(predicted_probs) 
#    entropy_array = entr(predicted_probs).sum(axis=1)
#    query_index, value = max(enumerate(entropy_array), key=operator.itemgetter(1))
#    print('The most uncertain point is No.', query_index)    # Calculate and report our model's accuracy.
#    return unqueried_score, query_index, predictions

    