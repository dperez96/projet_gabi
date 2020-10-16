from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import numpy.random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from Constants import *
import matplotlib.pyplot as plt


def prepare_test_train(X_train, X_test):  # (X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    """
    Normalisation of the training and testing data
    """
    sc = StandardScaler()
    if len(np.shape(X_train)) == 1:  # in case there is only 1 feature
        X_train = np.array(X_train).reshape(-1, 1)
        X_test = np.array(X_test).reshape(-1, 1)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test  # , y_train, y_test


def compute_score(y_pred, y_test):
    """
    The metric to score the performance of the classifiers is a weighted average of specificity and sensitivity
    This function computes this metric, based on the predictions of the classifier, compared to the actual classes
    """
    conf_mat = confusion_matrix(y_test, y_pred)  # the confusion matrix is a matrix where the element of the ith line
    # and jth column is the number of individuals of the ith class categorized by the classifier in the jth class
    specificity = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1])  # sp = TN/TN+FP
    sensitivity = float(conf_mat[1][1]) / (conf_mat[0][0] + conf_mat[1][0])  # sen = TP/TP+FN
    score = 0.66 * sensitivity + 0.33 * specificity
    return score


def random_forest_test(X_train, X_test, y_train, y_test, n_estimators=60, random_state=1, oob_score=False,
                       bootstrap=True, criterion='gini', max_depth=None, class_weight='balanced'):
    """
    Trains a random forest model with the hyper-parameters given as arguments, then computes the score of the classifier
    on the training and testing set
    """
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, oob_score=oob_score,
                                 bootstrap=bootstrap, criterion=criterion, max_depth=max_depth,
                                 class_weight=class_weight)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))
    train_score = rfc.score(X_train, y_train)
    test_score = compute_score(y_pred, y_test)
    # importances = rfc.feature_importances_
    # print('importances : ' + str(importances))
    return train_score, test_score


def log_reg_test(X_train, X_test, y_train, y_test):
    """
    Trains a logistic regression classifier, then computes and prints a report of the classification
    """
    regressor = LogisticRegression().fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    return regressor


def svc_test(X_train, X_test, y_train, y_test, C=1, kernel='rbf'):
    """
    Trains a random svm model with the hyper-parameters given as arguments, then computes the score of the classifier
    """
    # c = [0.1, 0.5, 1, 5, 10, 50, 100]
    # for elem in c:
    svc = SVC(C=C, kernel=kernel)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    score = compute_score(y_pred, y_test)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print('training_score = ' + str(svc.score(X_train, y_train)))
    # print('test accuracy = ' + str(accuracy_score(y_test, y_pred)))

    return score


def shuffle_in_unison(a, b):
    """
    function used in the crossvalidation to shuffle the X and y arrays in unison
    """
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def divide_in_k_samples(X, y, k):
    """
    function used in the crossvalidation to split the dataset in k subsets, where the balanced between classes is
    concerved
    """
    # the first step is to split the dataset into its two classes, in order to shuffle both classes separately to make
    # sure that there will be the same number of individuals of each class in the eventual k subsets
    X0 = X[:int(len(X) / 2)]
    y0 = y[:int(len(X) / 2)]
    X0, y0 = shuffle_in_unison(X0, y0)

    X1 = X[int(len(X) / 2):]
    y1 = y[int(len(X) / 2):]
    X1, y1 = shuffle_in_unison(X1, y1)

    samples_X = []  # these arrays contain the k subsets of X and y
    samples_y = []
    for i in range(k):
        Xk0 = X0[i * len(X0) // k: (i + 1) * len(X0) // k]  # the i/kth of X0 and X1 are selected
        Xk1 = X1[i * len(X1) // k: (i + 1) * len(X1) // k]

        yk0 = y0[i * len(y0) // k: (i + 1) * len(y0) // k]
        yk1 = y1[i * len(y1) // k: (i + 1) * len(y1) // k]
        if len(np.shape(Xk0)) > 1:
            Xk = np.zeros((len(Xk0) + len(Xk1), len(Xk0[0])))  # this array will contain the i/kth of X : Xk0 and Xk1
        elif len(np.shape(Xk0)) == 1:  # this is in the case there is only 1 feature computed
            Xk = np.zeros((len(Xk0) + len(Xk1)))
        yk = np.zeros((len(yk0) + len(yk1)))

        Xk[:len(Xk0)] = Xk0  # Xk0, Xk1, yk0, yk1 are added to Xk and yk
        Xk[len(Xk0):] = Xk1
        yk[:len(yk0)] = yk0
        yk[len(yk0):] = yk1

        samples_X.append(Xk)  # Xk and yk are added in samples_X and samples_y
        samples_y.append(yk)
    return samples_X, samples_y


def k_cross_val_svc(X, y, k=4, C=1, kernel='rbf'):
    """
    This function implements k-fold stratified crossvalidation on a svc model, with the hyper-parameters given in the
    argument
    """

    samples_X, samples_y = divide_in_k_samples(X, y, k)  # The dataset and the number of folds are given in the argument
    # and the function returns the same dataset split into k subsets
    scores = []
    for i in range(k):  # at each iteration of the loop, a different subset is used as testing data
        print('i = ' + str(i), end=' ')
        X_test = samples_X[i]
        y_test = samples_y[i]
        X_train, y_train = [], []
        for j in range(k):  # all the other subset are used as training data
            if j != i:
                X_train.extend(samples_X[j])  # the k-1 arrays of samples_X are added in one array (X_train) which has
                # the appropriate format to be given as input to "prepare_test_train"
                y_train.extend(samples_y[j])
        X_train, X_test = prepare_test_train(X_train, X_test)  # the data is normalized
        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        score = svc_test(X_train, X_test, y_train, y_test, C, kernel)  # the classifier is trained then evaluated
        scores.append(score)  # the k scores are kept in this array

    return scores


def k_cross_val_rdf(X, y, k=4, n_estimators=10, criterion='gini', max_depth=None):
    """
    This function implements k-fold stratified crossvalidation on a random forest model, with the hyper-parameters given
    in the argument
    """
    samples_X, samples_y = divide_in_k_samples(X, y, k)  # The dataset and the number of folds are given in the argument
    # and the function returns the same dataset split into k subsets
    train_scores, test_scores = [], []
    for i in range(k):  # at each iteration of the loop, a different subset is used as testing data
        print('i = ' + str(i), end=' ')
        X_test = samples_X[i]
        y_test = samples_y[i]
        X_train, y_train = [], []
        for j in range(k):  # all the other subset are used as training data
            if j != i:
                X_train.extend(samples_X[j])  # the k-1 arrays of samples_X are added in one array (X_train) which has
                # the appropriate format to be given as input to "prepare_test_train"
                y_train.extend(samples_y[j])
        X_train, X_test = prepare_test_train(X_train, X_test)  # the data is normalized
        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        train_score, test_score = random_forest_test(X_train, X_test, y_train, y_test, n_estimators=n_estimators,
                                                     criterion=criterion, max_depth=max_depth)  # the classifier is
        # trained then evaluated
        train_scores.append(train_score)  # the k scores are kept in this array
        test_scores.append(test_score)

    return train_scores, test_scores


def test_1_rdf():
    """
    training and evaluating one random forest model (with 3 datasets, corresponding to 3 selections of non apneic
    periods)
    """
    all_train_scores = []
    all_test_scores = []
    for s in range(3):
        print('s = ' + str(s), end=' ')
        X = np.genfromtxt('../features_data/X_' + str(s) + '.csv', delimiter=',')
        y = np.genfromtxt('../features_data/y_' + str(s) + '.csv', delimiter=',')
        train_scores, test_scores = k_cross_val_rdf(X, y, n_estimators=80, max_depth=12)
        all_train_scores.extend(train_scores)
        all_test_scores.extend(test_scores)
    print('\naverage train : ' + str(np.average(all_train_scores)) + ' +/- ' + str(np.std(all_train_scores)))
    print('average test : ' + str(np.average(all_test_scores)) + ' +/- ' + str(np.std(all_test_scores)))


def final_rdf_test():
    X_train = np.genfromtxt('../features_data/X_1_with_hypopnea.csv', delimiter=',')
    y_train = np.genfromtxt('../features_data/y_1_with_hypopnea.csv', delimiter=',')
    X_test = np.genfromtxt('../features_data/X_test_1_with_hypopnea.csv', delimiter=',')
    y_test = np.genfromtxt('../features_data/y_test_1_with_hypopnea.csv', delimiter=',')

    train_score, test_score = random_forest_test(X_train, X_test, y_train, y_test, n_estimators=80, random_state=1,
                                                 bootstrap=True, criterion='gini', max_depth=12,
                                                 class_weight={0: 10, 1: 3})
    print('train score : ' + str(train_score))
    print('test_score : ' + str(test_score))


def rdf_pr_curve(X_train, X_test, y_train, y_test, classif_type, n_estimators=80, random_state=1, max_depth=30):
    """
    Trains a random forest model with the hyper-parameters given as arguments, then computes the score of the classifier
    on the training and testing set
    """
    if classif_type == 'random_forest':
        print('ok!')
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, oob_score=False,
                                            bootstrap=True, criterion='gini', max_depth=max_depth,
                                            class_weight={0: 5, 1: 1})
    else:  # classif_type == 'gradient_boosting'
        classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=n_estimators,
                                                max_depth=max_depth, random_state=random_state)
    classifier.fit(X_train, y_train)
    pr = []
    for thresh in np.arange(0, 1, 0.02):
        # print('tresh = ' + str(thresh))

        probas = classifier.predict_proba(X_test)
        # print('probas = ' + str(probas))
        y_pred = np.array(probas[:, 1] > thresh)
        # print('ypred = ' + str(y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print('cm = ' + str(cm))
        if cm[1][1] == 0:
            if cm[0][1] == 0:
                precision = float('nan')
            elif cm[1][0] == 0:
                recall = float('nan')
            else:
                precision = recall = 0
        else:
            precision = float(cm[1][1]) / (cm[1][1] + cm[0][1])
            recall = float(cm[1][1]) / (cm[1][1] + cm[1][0])
        pr.append([recall, precision])
    return np.array(pr)


def make_pr(i):
    X_train = np.genfromtxt('../features_data/212_patients_final_tests/X_1_with_hypopnea.csv', delimiter=',')
    y_train = np.genfromtxt('../features_data/212_patients_final_tests/y_1_with_hypopnea.csv', delimiter=',')
    X_test = np.genfromtxt('../features_data/212_patients_final_tests/X_test_1_with_hypopnea.csv', delimiter=',')
    y_test = np.genfromtxt('../features_data/212_patients_final_tests/y_test_1_with_hypopnea.csv', delimiter=',')

    pr = rdf_pr_curve(X_train, X_test, y_train, y_test, n_estimators=80, random_state=1,
                      bootstrap=True, criterion='gini', max_depth=12, class_weight={0: 1, 1: i})
    plt.figure()
    plt.plot(pr[:, 0], pr[:, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('i = ' + str(i))
    return pr


def make_k_stratified_samples(ratio, k=4, without_hypopneas=1):
    name_a = '../features_data/all_features_a_train'
    name_wa = '../features_data/all_features_wa_train'
    if without_hypopneas:
        name_a += '_without_hypopnea'
        name_wa += '_without_hypopnea'
    name_a += '.csv'
    name_wa += '.csv'
    all_features_a = np.genfromtxt(name_a)
    all_features_wa = np.genfromtxt(name_wa)

    number_of_a = len(all_features_a)
    if number_of_a * ratio < len(all_features_wa):
        number_of_wa = round(number_of_a * ratio)
    else:
        number_of_wa = len(all_features_wa)
        number_of_a = int(number_of_wa // ratio)

    X = []
    y = []
    for i in range(k):
        Xk = []
        n_a = number_of_a // k
        # print('n_a : ' + str(n_a))
        n_wa = number_of_wa // k
        # print('n_wa : ' + str(n_wa))
        Xk.extend(all_features_wa[i * n_wa:(i + 1) * n_wa])
        Xk.extend(all_features_a[i * n_a:(i + 1) * n_a])
        yk = np.zeros(n_a + n_wa)
        yk[n_wa:] = 1
        X.append(Xk)
        y.append(yk)
    return np.array(X), np.array(y)


def make_test_x_y(without_hypopneas=1):
    name_a = '../features_data/all_features_a_test'
    name_wa = '../features_data/all_features_wa_test'
    if without_hypopneas:
        name_a += '_without_hypopnea'
        name_wa += '_without_hypopnea'
    name_a += '.csv'
    name_wa += '.csv'
    all_features_a = np.genfromtxt(name_a)
    all_features_wa = np.genfromtxt(name_wa)

    X = np.ndarray((len(all_features_a) + len(all_features_wa), len(all_features_a[0])))
    y = np.zeros(len(all_features_a) + len(all_features_wa))

    X[:len(all_features_wa)] = all_features_wa
    X[len(all_features_wa):] = all_features_a
    y[len(all_features_wa):] = 1

    return X, y


def stratified_cross_val_rdf(classifier_type, ratio=45, k=1):
    X, y = make_k_stratified_samples(ratio, k)

    all_pr = []
    for i in range(k):
        print('i = ' + str(i))
        X_train = X[i]
        y_train = y[i]
        X_test, y_test = make_test_x_y()

        pr = rdf_pr_curve(X_train, X_test, y_train, y_test, classifier_type)
        all_pr.append(pr)
    if k == 1:
        av_pr = all_pr[0]
        print('av_pr : ' + str(av_pr))
    else:
        av_pr = np.average(all_pr, axis=0)

    max_pr_score = 0
    index = 0
    index_max = 0
    for point in av_pr:
        index += 1
        # pr_score = ((point[0] ** 2 + point[1] ** 2) / 2) ** (1 / 2)
        pr_score = (point[0] * point[1]) ** (1 / 2)
        # print('point : ' + str(point))
        # print('pr_score : ' + str(pr_score))
        if pr_score > max_pr_score:
            max_pr_score = pr_score
            recall = point[0]
            precision = point[1]
            index_max = index
    print('recall : ' + str(recall))
    print('precision : ' + str(precision))
    print('threshold = ' + str(0.04 + index_max * 0.02))

    fig = plt.figure()
    plt.plot(av_pr[:, 0], av_pr[:, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    #plt.suptitle('200 estimators')
    plt.title('best score : ' + str(round(max_pr_score, 2)) + '\nwith recall = ' + str(
        round(recall, 2)) + ' and precision = ' +
              str(round(precision, 2)))
    #plt.text(0.3, 0.3, 'loss=\'deviance\', \nlearning_rate=0.1, \nn_estimators=80, \nmax_depth =  17, \nratio = 4')
    plt.text(0.2, 0.2, 'ratio = ' + str(ratio)+', \n80 estimators, \nmax depth=30, \nclass_weight = 5-1')
    return max_pr_score, av_pr
