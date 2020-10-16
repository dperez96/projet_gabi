import pyedflib as pyedflib
import numpy as np
import utils
from time import time
import pandas as pd
from os import listdir
from Constants import *


def get_ids(id_path):
    """
    :param id_path: string of the path of the directory with the edf files
    :return: a list of the ids of all edf files in the id_path directory
    """
    ids = listdir(id_path)
    new_ids = []
    for elem in ids:
        new_ids.append(elem[-8:-4])  # selecting the part of the string corresponding to the id
    return new_ids


def load_raw_edf(file_id, signal_name):
    """
    load raw data from an edf file, selects one signal and returns it as an np array
    :param file_id: string : the id of the recording to load
    :param signal_name: type of the signal of the recording. String : 'HR' or 'SpO2' for example
    :return: np array. The first item is a vector with the timestamps, and the second is the values of the signal
    """
    edf_file = pyedflib.EdfReader('../mesa_data/polysomnography/edfs/mesa-sleep-' + file_id + '.edf')
    signal_labels = edf_file.getSignalLabels()

    col_index = signal_labels.index(signal_name)  # selection of the index corresponding to the wanted signal

    sample_frequencies = edf_file.getSampleFrequencies()

    signal = edf_file.readSignal(col_index)
    sf = sample_frequencies[col_index]

    time_sig = np.array(range(0, len(signal)))  # Get timestamps for signal data
    time_sig = time_sig / sf

    data = np.transpose(np.vstack((time_sig, signal)))
    data = utils.remove_nans(data)
    return data


def load_raw(file_id, signal_type):
    """
    returns the signal from the specified txt file (as np array)
    :param file_id: string
    :param signal_type: string : 'HR' or 'SpO2' for example
    :return:np array. The first item is a vector with the timestamps, and the second is the values of the signal
    """
    file_name = '../mesa_data/polysomnography/new_txts/' + file_id + '-' + signal_type
    array = pd.read_csv(file_name, delimiter=' ')
    data = array.values  # data[0] is the time vector and data[1] the signal
    return data


def write_new_txt(t, sig, file_id, sig_name):
    """
    function to save a signal in a txt file
    :param t: array with timestamps
    :param sig: array with the signal values
    :param file_id: string. id of the recording of the signal (4 digits string)
    :param sig_name: string. type of signal ('HR' or 'SpO2' for example
    """
    file_name = '../mesa_data/polysomnography/new_txts/' + file_id + '-' + sig_name
    txt_file = open(file_name, 'w')
    for i in range(len(t) - 1):
        txt_file.write(str(t[i]) + ' ' + str(sig[i]) + '\n')
    txt_file.close()
    return 1


def edf_to_txt(signal_name):
    """
    writes txt files with specified signal from all edf files in a directory
    :param signal_name: string. 'HR' or 'SpO2' for example
    :return:
    """
    for file_id in FILE_IDs[-12:]:
        start = time()
        data = load_raw_edf(file_id, signal_name)
        t = data[:, 0]
        sig = data[:, 1]
        write_new_txt(t, sig, file_id, signal_name)
        print(str(file_id) + ' : ' + str(time() - start) + ' s')
        return 1


def test_hr_spo2():
    """
    checks if for each individual, the hr and spo2 signals have the same length
    :return: 1 if it is the case, 0 if the lengths are different
    """
    ok = 1
    for file_id in FILE_IDs:
        print(str(FILE_IDs.index(file_id) + 1) + '/' + str(len(FILE_IDs)))
        hr = load_raw(file_id, 'HR')
        SpO2 = load_raw(file_id, 'SpO2')
        if len(hr[0]) != len(SpO2[0]):
            print(file_id + '!!!')
            ok = 0
    if ok:
        print('all hr and spo2 signals are compatible')
    return ok
