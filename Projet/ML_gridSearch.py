import pandas as pd
import numpy as np
import sys
import timeit
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing, svm, neighbors, tree, neural_network
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer

FLOATPRECISION = 3
CROSSVALIDATION = 3
PROJECTPATH = "/home/david/Documents/HEPIA/hepia_19_20/ML/machine_learning/Projet/"

def writeOutput(output, filename):
    with open(PROJECTPATH + filename +'.csv', 'w') as csvFile:
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
   
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    print(len(means), "models computed in", round(stop-start, 2), "seconds", 
                                "with parameters", *clf.cv_results_['params'][0])
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        outPut.append([round(mean, FLOATPRECISION), round(std * 2, FLOATPRECISION), *params.values()])
    
    return outPut


dataWisc = np.loadtxt(PROJECTPATH + "wisconsin.dat", dtype = int, delimiter=',', skiprows = 14)

dowJonesdf = pd.read_csv(PROJECTPATH + "dow_jones_index/dow_jones_index.data", sep=",")

#dowJonesdf['date'] = pd.to_datetime(dowJonesdf['date'])

y = dowJonesdf['percent_change_next_weeks_price'] 

le = preprocessing.LabelEncoder()

y = le.fit_transform(y) # transform to int
#print(y)

feature_cols = ['percent_change_price', 'percent_change_volume_over_last_wk', 
                    'days_to_next_dividend', 'percent_return_next_dividend']

X = dowJonesdf.loc[:, feature_cols]


#print(X)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_dj = imp.fit(X)

X_imp  = imp_dj.transform(X)
X_scaled = preprocessing.scale(X_imp)

est = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
est.fit(X_imp)
Xt = est.transform(X_imp)
'''
estY = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
estY.fit(y)
Yt = estY.transform(y)
'''
#scores = cross_val_score(neigh, X_scaled, y, cv=CROSSVALIDATION)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
#print("score DJ", scores)

neigh = neighbors.KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

print(Xt)
print("score", neigh.score(X_test, y_test))
print(y)
outputList = []

if "SVM" in sys.argv[1:]:
    tuned_parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma' : [0.1, 1, 10, 100],
        'C' : [0.1, 1, 10, 100, 1000],
        'degree' : [0, 1, 2, 3, 4, 5, 6],
    }
    svm_wisc = computeScore(dataWisc, svm.SVC(), tuned_parameters)
    outputList.append(svm_wisc,"svm_wisc")


if "PPV" in sys.argv[1:]:
    n_neighbors = list(range(1, 50, 2))
    
    tuned_parameters = {
        'n_neighbors' : n_neighbors,
        'weights' : ['uniform', 'distance']
    }
    ppv_wisc =  computeScore(dataWisc, neighbors.KNeighborsClassifier(), tuned_parameters)
    outputList.append((ppv_wisc, "ppv_wisc"))


if "AD" in sys.argv[1:]:
    range_min = list(range(2, 20))
    max_depth = list(range(1,100))
    tuned_parameters = {
        'max_depth': max_depth,
        'min_samples_leaf' : range_min,
        'min_samples_split' : range_min
    }
    ad_wisc = computeScore(dataWisc, tree.DecisionTreeClassifier(), tuned_parameters)
    outputList.append((ad_wisc, "ad_wisc"))


if "PMC" in sys.argv[1:]:
    tuned_parameters = {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'alpha': [0.1, 1, 10],
        'hidden_layer_sizes': [(5, 50), (5, 100), (10, 50), (10,100)],
        'learning_rate' : ["constant", "invscaling", "adaptive"],
        'solver' : ['lbfgs', 'sgd', 'adam']
    }
    pmc_wisc = computeScore(dataWisc, neural_network.MLPClassifier(), tuned_parameters)
    outputList.append((pmc_wisc, "pmc_wisc"))

for outPut, filename in outputList:
    writeOutput(outPut, filename)