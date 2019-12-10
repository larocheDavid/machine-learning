import pandas as pd
import numpy as np
import sys
import timeit
import csv
import os
from sklearn import preprocessing, svm, neighbors, tree, neural_network
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


FLOATPRECISION = 3
CROSSVALIDATION = 3
CWD = os.getcwd()


def writeOutput(output, filename):

    newpath = CWD + '/output/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open(newpath + filename + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(output)
    csvFile.close()


def computeScore(data, classifier, tuned_parameters):
    X, y = data[:,:-1], data[:,-1]
    
    outPut = [['TCC moyen test', 'STD', 'TCC Train Moyen', 
                    'Fit Time Moyen'] + [key.replace('_', ' ').title() for key in tuned_parameters]]

    start = timeit.default_timer()
    clf = GridSearchCV(classifier, tuned_parameters, cv=CROSSVALIDATION, scoring='accuracy', n_jobs=-1, return_train_score=True)
    clf.fit(X, y)
    stop = timeit.default_timer()
    means = clf.cv_results_['mean_test_score']

    print(len(means), 'models computed in', round(stop-start, 2), 'seconds', 
                                'with parameters', *clf.cv_results_['params'][0])
    for mean, std, mean_train, fit_time, params in zip(
            means, clf.cv_results_['std_test_score'], clf.cv_results_['mean_train_score'],
            clf.cv_results_['mean_fit_time'], clf.cv_results_['params']):

        outPut.append([round(mean, FLOATPRECISION), round(std * 2, FLOATPRECISION), 
                            round(mean_train, FLOATPRECISION), round(fit_time, FLOATPRECISION +1), *params.values()])
   
    return outPut


def cleanDowJones():
    
    dowJonesdf = pd.read_csv(CWD + '/data/dow_jones_index.data', sep=',')

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


dataWisc = np.loadtxt(CWD + '/data/wisconsin.dat', dtype = int, delimiter=',', skiprows = 14)

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
    
    tuned_parameters = {
        #'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors' : [i for i in range(1, 50, 2)],
        'weights' : ['uniform', 'distance'],
    }
    if 'wisc' in sys.argv[1:]:
        wisc_ppv = computeScore(dataWisc, neighbors.KNeighborsClassifier(), tuned_parameters)
        outputList.append((wisc_ppv, 'wisc_ppv'))

    if 'dj' in sys.argv[1:]:
       dj_ppv = computeScore(dataDowJones, neighbors.KNeighborsClassifier(), tuned_parameters)
       outputList.append((dj_ppv, 'dj_ppv'))


if 'AD' in sys.argv[1:]:
    tuned_parameters = {
        'max_depth': [i for i in range(1, 100)],
        'min_samples_leaf' : [i for i in range(2, 20)],
        'min_samples_split' : [i for i in range(2, 20)]
    }
    if 'wisc' in sys.argv[1:]:
        wisc_ad = computeScore(dataWisc, tree.DecisionTreeClassifier(), tuned_parameters)
        outputList.append((wisc_ad, 'wisc_ad'))
    
    if 'dj' in sys.argv[1:]:
        dj_ad = computeScore(dataDowJones, tree.DecisionTreeClassifier(), tuned_parameters)
        outputList.append((dj_ad, 'dj_ad'))

    if 'arcing' in sys.argv[1:]:
        hyper_parameters = {
            'max_depth' : [1],
            'n_estimators' : [i for i in range(1, 102, 10)],
        }
        dj_ad_arcing = computeScore(dataDowJones, RandomForestClassifier(), hyper_parameters)
        outputList.append((dj_ad_arcing, 'dj_ad_arcing'))


if 'PMC' in sys.argv[1:]:
    tuned_parameters = {
        #'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        #'alpha': [0.1, 1, 10],
        'hidden_layer_sizes': [50, 100, 200],#[(5, 50), (5, 100), (10, 50), (10,100)],
        'learning_rate' : ['constant', 'invscaling', 'adaptive'],
        #'solver' : ['lbfgs', 'sgd', 'adam']
    }
    if 'wisc' in sys.argv[1:]:
        wisc_pmc = computeScore(dataWisc, neural_network.MLPClassifier(), tuned_parameters)
        outputList.append((wisc_pmc, 'wisc_pmc'))
        
    if 'dj' in sys.argv[1:]:
        dj_pmc = computeScore(dataWisc, neural_network.MLPClassifier(), tuned_parameters)
        outputList.append((dj_pmc, 'dj_pmc'))

    if 'bagging' in sys.argv[1:]:
        hyper_parameters = {
            'base_estimator__hidden_layer_sizes': [50, 100, 200],#[i for i in range(1, 202, 25)],
            'base_estimator__learning_rate' : ['constant', 'invscaling', 'adaptive'],
            'max_samples' : [0.5, 0.6321, 0.9],
            'n_estimators' : [10, 15, 20],
        }
        wisc_pmc_bagging = computeScore(dataWisc, BaggingClassifier(neural_network.MLPClassifier()), hyper_parameters)
        outputList.append((wisc_pmc_bagging, 'wisc_pmc_bagging'))


for outPut, filename in outputList:
    writeOutput(outPut, filename)
