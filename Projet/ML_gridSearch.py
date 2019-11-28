import numpy as np
import sys
import timeit
import matplotlib.pyplot as plt
import csv
from sklearn import svm, neighbors, tree, neural_network
from sklearn.model_selection import cross_val_score, GridSearchCV

FLOATPRECISION = 3
CROSSVALIDATION = 3

def writeOutput(output, filename):
    with open(projectPath + filename +'.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(output)
    csvFile.close()


def computeScore(data, estimator, tuned_parameters):
    X, y = data[:,:-1], data[:,-1]
    
    outPut = [["Accuracy", "STD"] + [key for key in tuned_parameters]]
    start = timeit.default_timer()
    clf = GridSearchCV(estimator, tuned_parameters, cv=CROSSVALIDATION, scoring='accuracy', n_jobs=-1)
    clf.fit(X, y)
    stop = timeit.default_timer()
    print(len(clf.cv_results_), "models computed in", stop-start, "seconds")
   
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    print(*clf.cv_results_['params'][0])
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        outPut.append([round(mean, FLOATPRECISION), round(std * 2, FLOATPRECISION), *params.values()])
    
    return outPut


if __name__ == "__main__":
    projectPath = "/home/david/Documents/HEPIA/hepia_19_20/ML/machine_learning/Projet/"

    dataWisc = np.loadtxt(projectPath + "wisconsin.dat", dtype = int, delimiter=',', skiprows = 14)

    if "SVM" in sys.argv[1:]:
        tuned_parameters = {
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma' : [0.1, 1, 10, 100],
            'C' : [0.1, 1, 10, 100, 1000],
            'degree' : [0, 1, 2, 3, 4, 5, 6],
        }
        svm_output = computeScore(dataWisc, svm.SVC(), tuned_parameters)
        if "writeSVM" in sys.argv[1:]:
            writeOutput(svm_output, "svm_wisc")
   

    if "PPV" in sys.argv[1:]:
        n_neighbors = list(range(1, 50, 2))
        
        tuned_parameters = {
            'n_neighbors' : n_neighbors,
            'weights' : ['uniform', 'distance']
        }
        ppv_output =  computeScore(dataWisc, neighbors.KNeighborsClassifier(), tuned_parameters)
        if "writePPV" in sys.argv[1:]:
            writeOutput(ppv_output, "ppv_wisc")


    if "AD" in sys.argv[1:]:
        range_min = list(range(2, 20))
        max_depth = list(range(1,100))
        tuned_parameters = {
            'max_depth': max_depth,
            'min_samples_leaf' : range_min,
            'min_samples_split' : range_min
        }
        
        ad_output = computeScore(dataWisc, tree.DecisionTreeClassifier(), tuned_parameters)
       
        if "writeAD" in sys.argv[1:]:
            writeOutput(ad_output, "ad_wisc")
    

    if "PMC" in sys.argv[1:]:
        tuned_parameters = {
            'alpha': [1,10,0.1],
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'hidden_layer_sizes': [ (5, 50), (5, 100), (10, 50), (10,100)],
            'learning_rate' : ["constant", "invscaling", "adaptive"],
            'solver' : ['lbfgs', 'sgd', 'adam']
        }
        pmc_output = computeScore(dataWisc, neural_network.MLPClassifier(), tuned_parameters)
        if "writePMC" in sys.argv[1:]:
            writeOutput(pmc_output, "pmc_wisc")

   