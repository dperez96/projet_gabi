from matplotlib.mlab import psd
import numpy as np
import sklearn.metrics
import fct_events as fe
from Constants import *
from entropy import entropy
import time_service as ts
import preprocessing as pp


def low_freq_energy(sig):
    """
    computes the low frequency energy of a signal. The low frequency energy is the area under the power spectral density
    of the signal between frequency bounds (F2_MIN and MAX) set in the module 'Constants'
    :param sig: list or nparray
    :return: float
    """
    p, f = psd(sig, NFFT=256)
    lf = integrate(f, p, F2_MIN, F2_MAX)
    return lf


def very_low_freq_energy(sig):
    """
    computes the very low frequency energy of a signal. The low frequency energy is the area under the power spectral
    density of the signal between frequency bounds (F1_MIN and MAX) set in the module 'Constants'
    :param sig: list or nparray
    :return: float
    """
    p, f = psd(sig, NFFT=256)
    vlf = integrate(f, p, F1_MIN, F1_MAX)
    return vlf


def integrate(x, y, mini, maxi):
    # computes the area under a curve given as argument - between the bounds given as argument
    bounds = (np.where(x > mini)[0][0], np.where(x > maxi)[0][0])
    if len(x[bounds[0]:bounds[1]]) > 1:
        res = sklearn.metrics.auc(x[bounds[0]:bounds[1]], y[bounds[0]:bounds[1]])
        # / sklearn.metrics.auc(x, y) essayer sans normalisation
    else:
        res = 0
    return res


def min_val(sig):
    # print('len sig = ' + str(len(sig)))
    return min(sig)


def max_val(sig):
    return max(sig)


def average(sig):
    return np.average(sig)


def median(sig):
    return np.median(sig)


def stand_dev(sig):
    return np.std(np.array(sig))


def n_times_when_under_threshold(sig, baseline):
    # computes the number of times when SpO2 reaches a value below a given threshold
    n = 0
    for item in sig:
        if item - baseline < SPO2_THRESHOLD:
            n += 1
    return n


def approximate_entropy(sig):
    res = entropy.app_entropy(sig, order=3, metric='chebyshev')
    if np.isnan(res) or res == float('inf'):
        res = 0
    return res


def sample_entropy(sig):
    res = entropy.sample_entropy(sig, order=3, metric='chebyshev')
    if np.isnan(res) or res == float('inf'):
        res = 0
    return res


def mid_ep(window):
    middle = (window[1] - window[0]) / 2
    mid_ep_start = middle - EPOCH / 2
    mid_ep_stop = middle + EPOCH / 2
    middle_ep = (window[0] + mid_ep_start, window[0] + mid_ep_stop)
    return middle_ep


def compute_epoch_features(ep, hr, spo2, features, apnea_times, feature_val_a, feature_val_wa, apnea_index,
                           no_apnea_index, epochs):
    """
    function which for each epoch, checks if it contains an apnea and computes all the features on the epoch. It then
    updates the arrays feature_val_a feature_val_wa, apnea_index and no_apnea_index (for the definition of these arrays,
    see the function compute_features)
    :param ep: tuple with the start and stop timestamps of the epoch
    :param hr: nparray
    :param spo2: nparray
    :param features: list of strings
    :param apnea_times: list of tuples, with the start and stop timestamps of each apnea event
    :param feature_val_a: np.ndarray
    :param feature_val_wa: np.ndarray
    :param apnea_index: list
    :param no_apnea_index: list
    :param epochs: list of tuples
    :return: the function only modifies lists
    """
    middle_ep = mid_ep(ep)
    # the middle epoch is the part of the window being
    # classified -- so where the presence of disturbed breathing should be checked

    contains_apnea = fe.contains_event(apnea_times, middle_ep)
    # contains_apnea is a boolean indicating there is an occurence of apnea during the epoch
    features_ep = []
    for feature in features:
        sig_type = feature[1]  # features can be computed on either HR or SpO2
        feature_type = feature[0]

        if sig_type == 'HR':  # computing the feature
            feature_val = feature_type(hr[ep[0]:ep[1]])
        else:  # sig_type == 'SpO2'
            if feature_type == n_times_when_under_threshold:  # for this feature, the baseline spo2 has to also be
                # given in the input
                baseline_spo2 = np.average(np.array(spo2))
                feature_val = feature_type(spo2[ep[0]:ep[1]], baseline_spo2)
            else:
                feature_val = feature_type(spo2[ep[0]:ep[1]])
        features_ep.append(feature_val)

    if contains_apnea:
        # depending on contains_apnea, the value will be added to the appropriate array.
        # the item of the corresponding index is removed from the other array
        feature_val_a[epochs.index(ep)] = features_ep
        apnea_index.append(epochs.index(ep))
    else:
        feature_val_wa[epochs.index(ep)] = features_ep
        no_apnea_index.append(epochs.index(ep))
    return 1


def compute_features(file_id, epochs, hr, spo2, features):
    """
    compute the features given in argument as tuple (signal_type,feature_type) of signals given as argument, for
    each period given as argument

    feature_val_a, feature_val_wa are arrays with the values of the computed features - each row correspond to an epoch,
    and each column to the values of the features computed on this epoch. the first array is for epochs containing
    apnea, and the 2nd for epochs without apnea
    :param file_id: 4 digits string
    :param epochs: list of tuples
    :param hr: list
    :param spo2: list
    :param features: list of strings
    :return: np.ndarrays
    """
    feature_val_a, feature_val_wa = np.zeros((len(epochs), len(features))), np.zeros((len(epochs), len(features)))
    # because we don't know yet the size of feauture_val_a and feature_val_wa, they are set as np arrays of maxiumum
    # size (the total number of epochs).
    # Let's take feature_val_a : each time an epoch is identified as containing an apnea, the array of features computed
    # on the window around this epoch will replace an array of zeros in feature_val_a. Its index will correspond to the
    # index of the window in 'epochs'. After all features are computed, we remove the arrays of zeros in
    # features_val_a which have not been modified (arrays at indexes corresponding to epochs without apneas)
    apnea_times = fe.prepare_events_data(file_id, APNEA_EVENTS)
    apnea_index = []  # list of indexes of epochs containing apneas
    no_apnea_index = []
    for ep in epochs:
        compute_epoch_features(ep, hr, spo2, features, apnea_times, feature_val_a, feature_val_wa, apnea_index,
                               no_apnea_index, epochs)

    feature_val_wa = feature_val_wa[no_apnea_index]
    # the items with indexes corresponding to epochs with apneas are removed from feature_val_wa
    feature_val_a = feature_val_a[apnea_index]

    return feature_val_a, feature_val_wa


def make_feature_all(features, train_test):
    """
    for the features given as argument, computes the features on all epochs of all people included in FILE_IDs (in the
    module 'Constants')

    all_feature_a, all_feature_wa are arrays with the values of the computed features - each row correspond to an epoch,
    and each column to the values of the features computed on this epoch. the first array is for epochs containing
    apnea, and the 2nd for epochs without apnea

    :param features: list of features (strings)
    :param train_test: boolean. Equals 1 if the features are computed on the training set, and 0 for the validation set.
    :return: nparrays
    """
    all_feature_a, all_feature_wa = [], []  # a = apnea, wa = without apnea
    print('patient ', end=' ')
    if train_test:
        file_ids = FILE_IDs  # FILE_IDs and TEST_FILE_IDs are defined in the module 'Constants'
    else:
        file_ids = TEST_FILE_IDs
    for file_id in file_ids:
        print(str(file_ids.index(file_id) + 1) + '/' + str(len(file_ids)), end=' ')
        raw_t_hr, raw_hr, raw_spo2, t_sig, hr, spo2 = pp.patient_sig(file_id)
        # computing the raw and processed hr and spo2 signals
        epochs = ts.new_sliding_window_array(t_sig, WINDOW, EPOCH)
        # computing the windows on which the features will be computed
        # (list of tuples with index of start and stop of each window)
        feat_a, feat_wa = compute_features(file_id, epochs, hr, spo2, features)
        # each element is an epoch, each epoch is an array with each feature
        for item in feat_a:  # the features computed on each individual are added in a list
            all_feature_a.append(item)
        for item in feat_wa:
            all_feature_wa.append(item)
    print('\nlen all_features_a = ' + str(len(all_feature_a)))
    print('len all_features_wa = ' + str(len(all_feature_wa)))
    return all_feature_a, all_feature_wa


def normalize(vec):
    """
    returns the normalized version of the vector given as argument (with mean = 0, std deviation = 1)
    """
    av = np.average(vec)
    std = np.std(vec)
    vec -= av
    vec = vec / std
    return vec


def normalize_array(array_a, array_wa):
    """
    array_a and array_wa are put in a single array. Each of its columns are then normalized
    ! there is already a normalization of the dataset in the classification_service module!
    """
    array = np.zeros((len(array_a[:, 0]) + len(array_wa[:, 0]), len(array_a[0])))
    array[:len(array_a[:, 0])] = array_a
    array[len(array_a[:, 0]):len(array_a[:, 0]) + len(array_wa[:, 0])] = array_wa
    for i in range(len(array[0])):
        array[:, i] = normalize(array[:, i])
    new_array_a = array[:len(array_a)]
    new_array_wa = array[len(array_a):]
    return new_array_a, new_array_wa


# list of features/signal types used in the classification
FEATURES = [(low_freq_energy, 'HR'), (low_freq_energy, 'SpO2'), (very_low_freq_energy, 'HR'),
            (very_low_freq_energy, 'SpO2'), (min_val, 'HR'), (min_val, 'SpO2'), (max_val, 'HR'),
            (stand_dev, 'HR'), (stand_dev, 'SpO2'), (sample_entropy, 'HR'),
            (sample_entropy, 'SpO2')]
