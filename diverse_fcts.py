import time_service as ts
from matplotlib.mlab import psd
from Constants import *
import preprocessing as pp
import numpy as np
import fct_mesa_data as fmd
import features_service as fs
from time import time
import random
import dataset_fcts as df
import classification_service as cs
import classif_tests as ct
import matplotlib.pyplot as plt


def check_sat_under_65():
    """
    checks the number of occurrences of spo2 being under 65% or varying of more than 4% in 1 sec
    checks the cumulated duration of these events and computes the average per person
    """

    u_65 = []  # array with the number of occurences of saturation under 65% for each person
    o_4_per_sec = []  # array with the number of occurences of saturation dropping of at least 4%/sec for each person
    for file_id in FILE_IDs:
        print(str((FILE_IDs.index(file_id) + 1)) + '/' + str(len(FILE_IDs)), end=' ')
        raw_t, raw_sig, t, sig = pp.one_patient_sig(file_id, 'SpO2')  # loading and filtering of the spo2 signal
        sat_under_65 = []  # array with the timestamps of all the occurences of saturation under 65% for this person
        var_above_4 = []
        for i in range(len(t)):
            if sig[i] < 65:
                sat_under_65.append(i)
            if i > 0 and sig[i] > (sig[i - 1] + 4):
                var_above_4.append(i)
        u_65.append(len(sat_under_65))
        o_4_per_sec.append(len(var_above_4))

    u_65 = np.array(u_65)
    o_4_per_sec = np.array(o_4_per_sec)
    print('average time with SpO2 under 65% per individual : ' + str(np.average(u_65)))
    print('average time with SpO2 varying of more than 4% in 1 sec : ' + str(np.average(o_4_per_sec)))
    u_65 = u_65[np.nonzero(u_65)]  # only item > zeros are kept in the array (--> length of the array = number of people
    o_4_per_sec = o_4_per_sec[np.nonzero(o_4_per_sec)]  # with at least one occurence of spo2 under 65/ drop of >4%
    print('number of individuals with occurences of SpO2 under 65% : ' + str(len(u_65)))
    print('number of individuals with occurences variation above 4%/sec : ' + str(len(o_4_per_sec)))

    print('average time under 65% for these individuals : ' + str(np.average(u_65)))
    print('average time under of variation above 4%/sec for these individuals : ' + str(np.average(o_4_per_sec)))


def new_txts_zero_extracted():
    """
    saves the cropped signals (with zeros removed) in new txt files
    """
    for file_id in FILE_IDs:
        raw_t, raw_hr, t, hr = pp.patient_sig(file_id)
        new_hr = []
        for i in range(len(raw_t)):
            if raw_t[i] in t:  # only index of times that were kept in the cropped signal are appended to new_hr
                new_hr.append(raw_hr[i])
        fmd.write_new_txt(t, new_hr, file_id, 'HR')  # new_hr and t are written in a txt file


def make_and_normalize(train_test='train'):
    """
    this function can be used to compute the features and then normalize them
    ! there is already a normalization of the dataset in the classification_service module! --> the normalize line is
    commented
    """
    features_list = fs.FEATURES
    all_feature_a, all_feature_wa = fs.make_feature_all(features_list, train_test)
    all_feature_a = np.array(all_feature_a)
    all_feature_wa = np.array(all_feature_wa)
    # all_feature_a, all_feature_wa = fs.normalize_array(all_feature_a, all_feature_wa)
    return all_feature_a, all_feature_wa


def prepare_training_data(seed):
    """
    This function is used for the computation of the features. After the computation, it selects a subset of the epochs
    without apnea in order to have as many epochs of both categories. The epochs are selected randomly, with a seed
    given in the argument.
    """
    top = time()
    all_feature_a, all_feature_wa = make_and_normalize()  # computation of the features

    y = np.float32(np.zeros(len(all_feature_a)+len(all_feature_a)))  # setting the X and y arrays
    y[:len(all_feature_a)] = 1  # y is the truth table for X (which contains the dataset given as input of the
    # classifier
    X = np.float32(np.zeros((len(all_feature_a)+len(all_feature_a), len(all_feature_a[0]))))

    X[:len(all_feature_a)] = all_feature_a
    random.seed(seed)
    random_wa_index = random.sample(range(len(all_feature_wa)), len(all_feature_a))  # selection of a subset of epochs
    # without apnea
    X[len(all_feature_a):len(all_feature_a) + len(all_feature_a)] = all_feature_wa[random_wa_index]
    print('time : ' + str(time() - top))
    return X, y


def prepare_train_test_data(seed):
    """
    Computation of the features, then the data is split between a training set and a testing set.
    For the training set, a subset of the periods without apneas is randomly selected in order to have balanced classes
    The training and testing sets are computed on different individuals
    """
    top = time()
    all_feature_a_train, all_feature_wa_train = make_and_normalize()  # computation of the features

    # setting the X and y_train array_trains
    y_train = np.float32(np.zeros(len(all_feature_a_train) + len(all_feature_a_train)))
    # y_train is the truth table for X (which contains the dataset given as input of the
    y_train[:len(all_feature_a_train)] = 1
    # classifier
    X_train = np.float32(np.zeros((len(all_feature_a_train) + len(all_feature_a_train), len(all_feature_a_train[0]))))

    X_train[:len(all_feature_a_train)] = all_feature_a_train
    random.seed(seed)
    # selection of a subset of epochs without apnea
    random_wa_index = random.sample(range(len(all_feature_wa_train)), len(all_feature_a_train))

    X_train[len(all_feature_a_train):len(all_feature_a_train) + len(all_feature_a_train)] = \
        all_feature_wa_train[random_wa_index]
    X_train[:len(all_feature_a_train)] = all_feature_a_train
    random.seed(seed)
    # selection of a subset of epochs without apnea
    random_wa_index = random.sample(range(len(all_feature_wa_train)), len(all_feature_a_train))
    X_train[len(all_feature_a_train):len(all_feature_a_train) + len(all_feature_a_train)] = \
        all_feature_wa_train[random_wa_index]

    all_feature_a_test, all_feature_wa_test = make_and_normalize('test')
    y_test = np.float32(np.zeros(len(all_feature_a_test) + len(all_feature_wa_test)))  # setting the X and y arrays
    y_test[:len(all_feature_a_test)] = 1  # y is the truth table for X (which contains the dataset given as input of the
    # classifier
    X_test = np.float32(np.zeros((len(all_feature_a_test) + len(all_feature_wa_test), len(all_feature_a_test[0]))))
    X_test[:len(all_feature_a_test)] = all_feature_a_test
    X_test[len(all_feature_a_test):] = all_feature_wa_test
    print('time : ' + str(time() - top))
    save_data(X_train, y_train, seed)
    save_data(X_test, y_test, seed, 'test')
    return 1


def prepare_stratified_training_data():
    top = time()
    all_feature_a, all_feature_wa = make_and_normalize()  # computation of the features
    random.shuffle(all_feature_wa)
    y = np.float32(np.zeros(len(all_feature_a) + len(all_feature_wa)))  # setting the X and y arrays
    y[:len(all_feature_a)] = 1  # y is the truth table for X (which contains the dataset given as input of the
    # classifier
    X = np.float32(np.zeros((len(all_feature_a) + len(all_feature_wa), len(all_feature_a[0]))))
    X[:len(all_feature_a)] = all_feature_a
    X[len(all_feature_a):] = all_feature_wa
    print('time : ' + str(time() - top))
    save_data(X, y)


def save_data(X, y, seed=None, train_test='train'):
    """
    function that saves the data given as argument in csv files
    the argument of the function also contains info required for the naming of the csv files
    """
    if train_test == 'train':
        np.savetxt('features_data/X_' + str(seed) + '_with_hypopnea.csv', X, delimiter=',')
        np.savetxt('features_data/y_' + str(seed) + '_with_hypopnea.csv', y, delimiter=',')
    else:  # train_test == 'test'
        np.savetxt('features_data/X_test_' + str(seed) + '_with_hypopnea.csv', X, delimiter=',')
        np.savetxt('features_data/y_test_' + str(seed) + '_with_hypopnea.csv', y, delimiter=',')


def save_features(all_features_a, all_features_wa):
    random.shuffle(all_features_wa)

    np.savetxt('features_data/all_features_a_without_hypopnea.csv', all_features_a)
    np.savetxt('features_data/all_features_wa_without_hypopnea.csv', all_features_wa)


def correlated_features(features):
    """
    This function computes the Pearson correlation coefficient between each pair of features
    It returns a matrix where the item on the ith line and jth column is the correlation between the ith and jth
    features
    It also prints the indices of the pair of features with a correlation coefficient abobe 0.95
    """
    feat_a, feat_wa = fs.make_feature_all(features)  # computation of the features
    X = np.zeros((len(feat_a)+len(feat_wa), len(feat_a[0])))  # X is an array with the features computed on all the
    # epochs
    X[:len(feat_a)] = feat_a
    X[len(feat_a):len(feat_a) + len(feat_wa)] = feat_wa
    corr_matrix = np.corrcoef(np.transpose(X))  # computation of the correlation matrix
    for i in range(len(corr_matrix)):  # these loops check for all the items in the matrix, and print the indices of the
        # pair of features with a correlation coefficient above 0.95
        for j in range(i, len(corr_matrix)):
            if i != j and corr_matrix[i][j] > 0.95:
                print('i-j = ' + str(i) + ' - ' + str(j))
                print('corr = ' + str(corr_matrix[i][j]))
    return corr_matrix



#t = time()
#score, av_pr = cs.stratified_cross_val_rdf('random_forest')

#print('time : ' + str(time() - t))

#all_ids = fmd.get_ids('mesa_data/polysomnography/edfs')
#print(all_ids)
#for sig in SIG_TYPES:
#    fmd.edf_to_txt(sig)

#prepare_train_test_data(1)
#print('ok')

#cs.test_1_rdf()

#ct.rfc_parameters_test()

#for sig in ['HR', 'SpO2']:
#     fmd.edf_to_txt(sig)
#t = time()
#for i in range(1, 5):
#    pr = cs.make_pr(i)
#print('time : ' + str(time() - t))
