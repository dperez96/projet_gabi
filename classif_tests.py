import numpy as np
import classification_service as cs
import Constants

"""
This module contains functions used for different varied tests, which mostly consists of loops where a classifier will 
be trained and evaluated on each iteration, with some parameter changing at each iteration. 

In the cases of svc/rfc_parameters_tests, the parameters tested are hyper-parameters of a svm/random forest model.
In the case of hr_smoothing_test, the parameter is the smoothing of the hr/spo2 signal. 
In the case of different_windows_test, the parameter is the size of the window on which the features are computed
"""


def crossval_test_k():
    """
    This function was used to evaluate the impact of the number of folds in the k-folds crossvalidation on the
    classification scoring and its variability
    """
    for k in range(5, 10):
        print('\nk : ' + str(k))
        all_scores = []
        for s in range(3):  # this loop is on the 3 selections of epochs that was made due to the class imbalance
            print('s = ' + str(s), end=' ')
            X = np.genfromtxt('../features_data/X_' + str(s) + '.csv', delimiter=',')  # loading of the features saved in
            # a csv file
            y = np.genfromtxt('../ffeatures_data/y_' + str(s) + '.csv', delimiter=',')
            scores = cs.k_cross_val_svc(X, y, k=k)  # computation of the cross-validation.
            all_scores.extend(scores)  # the scores for the different epoch selections and the k folds are all kept in
            # 1 array (so 3*k scores)
        print(all_scores)
        print('average : ' + str(np.average(all_scores)))
        print('std : ' + str(np.std(all_scores)))


def svc_parameters_test():
    """
    This function was made to evaluate the best set of hyper-parameters for the svc model
    """
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    Cs = [40, 50, 60]
    for kernel in kernels:
        for C in Cs:
            print('C = ' + str(C))
            print('kernel = ' + str(kernel))
            all_scores = []  # this array contains the scores of the k tests of the crossvalidation on the 3 datasets(s)
            # (selections of non apneic periods)
            for s in range(3):
                print('s = ' + str(s), end=' ')
                X = np.genfromtxt('../features_data/X_' + str(s) + '.csv', delimiter=',')  # loading of the data saved on
                y = np.genfromtxt('../features_data/y_' + str(s) + '.csv', delimiter=',')  # csv files
                scores = cs.k_cross_val_svc(X, y, C=C, kernel=kernel)
                all_scores.extend(scores)
            print(all_scores)
            print('average : ' + str(np.average(all_scores)))
            print('std : ' + str(np.std(all_scores)))


def rfc_parameters_test():
    """
    This function was made to evaluate the best set of hyper-parameters for the random forest model
    """
    n_estimators = [72, 73, 74, 75, 76, 77, 78]
    max_depth = [10]
    for md in max_depth:
        for n in n_estimators:
            print('max depth = ' + str(md))
            print('n_estimators = ' + str(n))
            all_train_scores, all_test_scores = [], []  # this array contains the scores of the k tests of the
            # crossvalidation on the 3 datasets(s) (selections of non apneic periods)
            for s in range(3):
                print('s = ' + str(s), end=' ')
                X = np.genfromtxt('../features_data/X_' + str(s) + '_no_hypopnea.csv', delimiter=',')  # loading of the data saved on
                y = np.genfromtxt('../features_data/y_' + str(s) + '_no_hypopnea.csv', delimiter=',')  # csv files
                train_scores, test_scores = cs.k_cross_val_rdf(X, y, n_estimators=n, max_depth=md)
                all_train_scores.extend(train_scores)
                all_test_scores.extend(test_scores)
            print('\naverage train : ' + str(np.average(all_train_scores)) + ' +/- ' + str(np.std(all_train_scores)))
            print('average test : ' + str(np.average(all_test_scores)) + ' +/- ' + str(np.std(all_test_scores)))


def hr_smoothing_test():
    for hrs in Constants.hr_smoothing:
        print('window size = ' + str(hrs))
        all_train_scores, all_test_scores = [], []
        for s in range(3):
            print('s = ' + str(s), end=' ')
            X = np.genfromtxt('../features_data/spo2_SG_smoothing/X_' + str(s) + '_hrs_' + str(hrs)
                              + '.csv', delimiter=',')
            y = np.genfromtxt('../features_data/spo2_SG_smoothing/y_' + str(s) + '_hrs_' + str(hrs)
                              + '.csv', delimiter=',')
            train_scores, test_scores = cs.k_cross_val_rdf(X, y, n_estimators=80, max_depth=12)
            all_train_scores.extend(train_scores)
            all_test_scores.extend(test_scores)
        print('\naverage train : ' + str(np.average(all_train_scores)) + ' +/- ' + str(np.std(all_train_scores)))
        print('average test : ' + str(np.average(all_test_scores)) + ' +/- ' + str(np.std(all_test_scores)))


def hr_raw_data_test():
    all_train_scores, all_test_scores = [], []
    for s in range(3):
        print('s = ' + str(s), end=' ')
        X = np.genfromtxt('../features_data/spo2_smoothing/X_' + str(s) + 'no_median' + '.csv', delimiter=',')
        y = np.genfromtxt('../features_data/spo2_smoothing/y_' + str(s) + 'no_median' + '.csv', delimiter=',')
        train_scores, test_scores = cs.k_cross_val_rdf(X, y, n_estimators=80, max_depth=12)
        all_train_scores.extend(train_scores)
        all_test_scores.extend(test_scores)
    print('\naverage train : ' + str(np.average(all_train_scores)) + ' +/- ' + str(np.std(all_train_scores)))
    print('average test : ' + str(np.average(all_test_scores)) + ' +/- ' + str(np.std(all_test_scores)))


def different_windows_test():
    for win in Constants.WINDOW:
        print('window size = ' + str(win))
        all_train_scores, all_test_scores = [], []
        for s in range(3):
            print('s = ' + str(s), end=' ')
            X = np.genfromtxt('../features_data/windows_tests/X_' + str(s) + 'win' + str(win) + '.csv', delimiter=',')
            y = np.genfromtxt('../features_data/windows_tests/y_' + str(s) + 'win' + str(win) + '.csv', delimiter=',')
            train_scores, test_scores = cs.k_cross_val_rdf(X, y, n_estimators=80, max_depth=12)
            all_train_scores.extend(train_scores)
            all_test_scores.extend(test_scores)
        print('\naverage train : ' + str(np.average(all_train_scores)) + ' +/- ' + str(np.std(all_train_scores)))
        print('average test : ' + str(np.average(all_test_scores)) + ' +/- ' + str(np.std(all_test_scores)))
