import pandas as pd
import numpy as np
import sys
import timeit
import csv
import os
from sklearn import preprocessing, svm, neighbors, tree, neural_network
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer

FLOATPRECISION = 3
CROSSVALIDATION = 10
PROJECTPATH = '/home/david/Documents/HEPIA/hepia_19_20/ML/machine_learning/Projet/'

def writeOutput(output, filename):

    newpath = PROJECTPATH + 'output/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open(newpath +filename +'.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(output)
    csvFile.close()


def computeScore(data, estimator, tuned_parameters):
    X, y = data[:,:-1], data[:,-1]
    
    outPut = [['mean_test_score', 'std_test_score', 'mean_train_score', 'mean_fit_time'] + [key for key in tuned_parameters]]
    start = timeit.default_timer()
    clf = GridSearchCV(estimator, tuned_parameters, cv=CROSSVALIDATION, scoring='accuracy', n_jobs=-1, verbose = 0, return_train_score=True)
    clf.fit(X, y)
    stop = timeit.default_timer()
   
    means = clf.cv_results_['mean_test_score']
    means_train = clf.cv_results_['mean_train_score']
    stds = clf.cv_results_['std_test_score']
    fit_times = clf.cv_results_['mean_fit_time']
    params =  clf.cv_results_['params']
    
    print(len(means), 'models computed in', round(stop-start, 2), 'seconds', 
                                'with parameters', *clf.cv_results_['params'][0])
    for mean, mean_train, std, fit_time, params in zip(means, means_train, stds, fit_times, params):
        outPut.append([round(mean, FLOATPRECISION), round(std * 2, FLOATPRECISION), round(mean_train, FLOATPRECISION), round(fit_time, FLOATPRECISION +1), *params.values()])
    
    return outPut


def cleanDowJones():
    
    dowJonesdf = pd.read_csv(PROJECTPATH + 'data/dow_jones_index.data', sep=',')

    feature_cols = ['quarter', 'open', 'high', 'low', 'close', 'volume', 'percent_change_price', 'percent_change_volume_over_last_wk', 'days_to_next_dividend']

    X = dowJonesdf[feature_cols].copy()

    for price in ['open', 'high', 'low', 'close']:
        X[price] = X[price].str[1:]

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X)
    X_imp = imputer.transform(X)
    X_scaled = preprocessing.scale(X_imp)

    y = np.where(dowJonesdf['percent_return_next_dividend'] > 1, 1, 0)

    return np.hstack((X_scaled, y.reshape(len(y), 1)))


dataWisc = np.loadtxt(PROJECTPATH + 'data/wisconsin.dat', dtype = int, delimiter=',', skiprows = 14)

dataDowJones = cleanDowJones()

outputList = []

if 'SVM' in sys.argv[1:]:
    tuned_parameters = {
        'C' : [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree' : [0, 1, 2, 3],
        'gamma' : [0.1, 1, 10],
    }
    if 'wisc' in sys.argv[1:]:
        wisc_svm = computeScore(dataWisc, svm.SVC(), tuned_parameters)
        outputList.append((wisc_svm, 'wisc_svm'))

    if 'dj' in sys.argv[1:]:
        dj_svm = computeScore(dataDowJones, svm.SVC(), tuned_parameters)
        outputList.append((dj_svm, 'dj_svm'))


if 'PPV' in sys.argv[1:]:
    n_neighbors = list(range(1, 50, 2))
    
    tuned_parameters = {
        #'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors' : n_neighbors,
        'weights' : ['uniform', 'distance'],
    }
    if 'wisc' in sys.argv[1:]:
        wisc_ppv = computeScore(dataWisc, neighbors.KNeighborsClassifier(), tuned_parameters)
        outputList.append((wisc_ppv, 'wisc_ppv'))

    if 'dj' in sys.argv[1:]:
       dj_ppv = computeScore(dataDowJones, neighbors.KNeighborsClassifier(), tuned_parameters)
       outputList.append((dj_ppv, 'dj_ppv'))


if 'AD' in sys.argv[1:]:
    range_min = list(range(2, 20))
    max_depth = list(range(1,100))
    tuned_parameters = {
        'max_depth': max_depth,
        'min_samples_leaf' : range_min,
        'min_samples_split' : range_min
    }
    if 'wisc' in sys.argv[1:]:
        wisc_ad = computeScore(dataWisc, tree.DecisionTreeClassifier(), tuned_parameters)
        outputList.append((wisc_ad, 'wisc_ad'))
    
    if 'dj' in sys.argv[1:]:
        dj_ad = computeScore(dataDowJones, tree.DecisionTreeClassifier(), tuned_parameters)
        outputList.append((dj_ad, 'dj_ad'))


if 'PMC' in sys.argv[1:]:
    tuned_parameters = {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'alpha': [0.1, 1, 10],
        'hidden_layer_sizes': [(5, 50), (5, 100), (10, 50), (10,100)],
        'learning_rate' : ['constant', 'invscaling', 'adaptive'],
        'solver' : ['lbfgs', 'sgd', 'adam']
    }
    if 'wisc' in sys.argv[1:]:
        wisc_pmc = computeScore(dataWisc, neural_network.MLPClassifier(), tuned_parameters)
        outputList.append((wisc_pmc, 'wisc_pmc'))
        
    if 'dj' in sys.argv[1:]:
        dj_pmc = computeScore(dataWisc, neural_network.MLPClassifier(), tuned_parameters)
        outputList.append((dj_pmc, 'dj_pmc'))

for outPut, filename in outputList:
    writeOutput(outPut, filename)
