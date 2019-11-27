import numpy as np
import timeit
import matplotlib.pyplot as plt
import csv
from sklearn import svm, neighbors, tree, neural_network
from sklearn.model_selection import cross_val_score

FLOATPRECISION = 2

def writeOutput(output, filename):
    with open(projectPath + filename +'.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(output)
    csvFile.close()

# SVM
def svmAlgo(data, kernels, gammas, cs, degrees): # output .csv file accuracy to hyperparameters

    X, y = data[:,:-1], data[:,-1]
    outPut = [["Kernel", "Gamma", "C", "Degree", "Run time", "Accuracy", "STD"]]

    for kernel in kernels:
        for gamma in gammas:
            for c in cs:
                for degree in degrees:
                    start = timeit.default_timer()
                    clf = svm.SVC(kernel=kernel, C=c, gamma=gamma, degree=degree) # SVM looking for largest minimum margin 
                    stop = timeit.default_timer()
                    scores = cross_val_score(clf, X, y, cv=10)
                    accuracy = round(scores.mean(), FLOATPRECISION)
                    std = round(scores.std() * 2, FLOATPRECISION)
                    outPut.append([kernel, gamma, c, degree, stop-start, accuracy,  std])

    return outPut

# K neighbors
def PPV(data, n_neighbors, weights):
    X, y = data[:,:-1], data[:,-1]
    outPut = [["K neighbors", "Weights", "Run time", "Accuracy", "STD"]]

    for n_neighbor in n_neighbors:
        for weight in weights:
            start = timeit.default_timer()
            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbor, weights=weight)
            stop = timeit.default_timer()
            scores = cross_val_score(clf, X, y, cv=10)#, scoring='accuracy')
            accuracy = round(scores.mean(), FLOATPRECISION)
            std = round(scores.std() * 2, FLOATPRECISION)
            outPut.append([n_neighbor, weight, stop-start, accuracy,  std])

    return outPut

# Decision Tree
def AD(data, max_depths, min_samples_splits, min_samples_leafs):
    X, y = data[:,:-1], data[:,-1]
    outPut = [["Max depth", "Min samples splits", "Min samples leafs", "Run time", "Accuracy", "STD"]]

    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                start = timeit.default_timer()
                clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, presort=True)
                stop = timeit.default_timer()
                scores = cross_val_score(clf, X, y, cv=10)
                accuracy = round(scores.mean(), FLOATPRECISION)
                std = round(scores.std() * 2, FLOATPRECISION)
                outPut.append([max_depth, min_samples_split, min_samples_leaf, stop-start, accuracy, std])

    return outPut

# Multilayer perceptron
def PMC(data, hidden_layer_sizes, activations, solvers, learning_rates):
    X, y = data[:,:-1], data[:,-1]
    outPut = [["Layers number", "Units number", "Activation", "Solver", "Learning rate", "Run time", "Accuracy", "STD"]]
    for activation in activations:
        for solver in solvers:
            for learning_rate in learning_rates:
                start = timeit.default_timer()
                clf = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, learning_rate=learning_rate)
                stop = timeit.default_timer()
                scores = cross_val_score(clf, X, y, cv=10)
                accuracy = round(scores.mean(), FLOATPRECISION)
                std = round(scores.std() * 2, FLOATPRECISION)
                outPut.append([hidden_layer_sizes[0], hidden_layer_sizes[1], activation, solver, learning_rate, stop-start, accuracy, std])

    return outPut

projectPath = "/home/david/Documents/HEPIA/hepia_19_20/ML/machine_learning/Projet/"

wisconsinData = np.loadtxt(projectPath + "wisconsin.dat", dtype = int, delimiter=',', skiprows = 14)

# SVM
'''
svm_output = svmAlgo(wisconsinData, kernels=['linear', 'rbf', 'poly'], gammas=[0.1, 1, 10, 100], cs=[0.1, 1, 10, 100, 1000], degrees=[0, 1, 2, 3, 4, 5, 6])
writeOutput(svm_output, "svm_output")
'''

# K neighbors
n_neighbors = list(range(1, 50, 2)) # distance weight more precise
ppv_output = PPV(wisconsinData, n_neighbors, weights = ['uniform', 'distance'])
writeOutput(ppv_output, "ppv_output")

# Decision Tree
max_depths = [30]#list(range(9, 50))
min_samples_splits = [5]#list(range(2, 10)) # The minimum number of samples required to split an internal node
min_samples_leafs =  [5]#list(range(1, 10))
ad_output = AD(wisconsinData, max_depths, min_samples_splits, min_samples_leafs)
writeOutput(ad_output, "ad_output")

# Multilayer perceptron
n_layers = 5
n_units = 100
hidden_layer_sizes = (n_layers, n_units)
activations = ["identity", "logistic", "tanh", "relu"]
solvers = ["lbfgs", "sgd", "adam"]
learning_rates = ["constant", "invscaling", "adaptive"]
pmc_output = PMC(wisconsinData, hidden_layer_sizes, activations, solvers, learning_rates)
writeOutput(pmc_output, "pmc_output")