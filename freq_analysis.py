import numpy as np
from scipy import signal
import preprocessing as pp
import time_service as ts
from Constants import *


def psd_list(file_id):
    """
    takes one person as input, computes the psd on the hr signal for all epochs , and keeps them in a list
    """
    list_of_psd = []
    raw_t, raw_hr, t, hr = pp.patient_sig(file_id)  # loading and filtering of the hr signal
    epochs = ts.sliding_window_array(t, WINDOW, EPOCH)  # generating the array of timestamps for the sliding windows
    for epoch in epochs:
        p, f = psd(hr[epoch[0]:epoch[1]], NFFT=300)  # computation of the psd
        list_of_psd.append((f, p, (epoch[0], epoch[1])))
    return list_of_psd


def psd(x):
    f, pow_den = signal.periodogram(x)
    return f, pow_den

