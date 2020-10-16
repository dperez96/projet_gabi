import numpy as np
from Constants import *
import matplotlib.pyplot as plt


def make_epochs(t):
    """
    generates an array of epochs (expressed as tuples (start,stop)) it is not used in any function: it was replaced by
    new_sliding_window_array
    :param t: list or nparray. Time vector
    :return: list of tuples. The tuples have the timestamps of the beginning and the end of the epoch
    """
    epochs = []
    start = 0
    new_epoch = 1
    for i in range(len(t) - 1):
        if new_epoch:
            start = i
            new_epoch = 0
        else:
            if i - start == 100 or t[i + 1] - t[i] > 2:
                stop = i
                epochs.append((start, stop))
                new_epoch = 1
    return epochs


def sliding_window_array(t, ep_len, step):
    """
    generates an array of sliding windows.
    it is not used in any function it was replaced by new_sliding_window_array
    :param t: list or nparray
    :param ep_len: int
    :param step: int
    :return: list of tuples, where each tuple has the timestamps of the beginning and end of a window
    """
    win_array = []
    for i in np.arange(0, len(t) - ep_len, step):
        start = i
        stop = i + ep_len
        win_array.append((start, stop))
    return win_array


def new_sliding_window_array(t, ep_len, step):
    """
    generates an array of sliding windows. A window cannot contain a time where the signal is cropped
    ep_len is the window length, step is the step between windows
    :param t: list or nparray
    :param ep_len: int
    :param step: int
    :return: list
    """
    win_array = []
    index_start = 0

    while index_start < len(t) - ep_len - 1:  # the last window starts ep_len seconds before the end of t
        # print(index_start)
        window_ok = 1  # window ok is 1 if it doesn't overlap with a time where the signal is cropped
        for i in range(ep_len - 1):
            if t[index_start + i] != (t[index_start + i + 1] - 1):
                window_ok = 0  # if window_ok is set to 0, it won't be added to the array of windows
        if window_ok:
            win_array.append((index_start, index_start+ep_len))
            # print((index_start, index_start+ep_len))
        index_start += step
    return win_array


def next_point(t, start):
    # takes as argument a cropped t vector, and an integer. if the integer is not in t,
    # the function returns the smallest number in t larger than the integer
    st = start
    b = 1
    while b and st < t[-1]:
        st += 1
        if st in t:
            b = 0
    return st


def previous_point(t, stop):
    # returns the largest number in t smaller than the argument
    st = stop
    b = 1
    while b and st > 0:
        st -= 1
        if st in t:
            b = 0
    return st


def find_t_indexes(t, epoch):
    # takes a cropped t and an epoch as argument, returns the epoch corrected to be completely included in t
    # this function is not used - it was made useless by new_sliding_window_array
    start = epoch[0]
    stop = epoch[1]
    ep_in_p = 1
    if start in t:
        i1 = t.index(start)
    else:  # l'epoch commence a un moment qui a ete enleve de t
        nextp = next_point(t, start)  # on fait commencer l'epoch a la valeur de t suivante la plus proche
        i1 = t.index(nextp)
        if np >= stop:  # toute l'epoch n'est pas dans t
            ep_in_p = 0
    if ep_in_p:
        if stop in t:
            i2 = t.index(stop)
        else:
            i2 = t.index(previous_point(t, stop))

    if ep_in_p:
        res = (i1, i2)
    else:
        res = -1
    #  print('find_t_indexes : ' + str(time.time()-top) + ' s')
    return res


def test_sliding_window(raw_t, t, raw_hr, sig_type, file_id):
    # function used in order to verify that the windows generated by the new_sliding_window_array function
    # does leave out times where the signal is equal to 0
    epochs = new_sliding_window_array(t, WINDOW, EPOCH)
    other_t = np.zeros(int(raw_t[-1]))  # array of the length of the recording time
    # each item in other_t corresponding to a time included in one of the window is set to &, 0 otherwise
    for elem in epochs:
        start = int(t[elem[0]])
        stop = int(t[elem[1]])
        for i in np.arange(start, stop - 1):
            other_t[i] = 1

    fig, axs = plt.subplots(2, 1)  # the first plot is the raw signal. it can be compared to the 2nd plot, where
    # the zeros should coincide with the zeros of the signal of the 1st plot
    fig.suptitle('patient ' + file_id)
    axs[0].plot(raw_t, raw_hr)
    axs[0].set_title(sig_type)
    axs[1].plot(range(len(other_t)), other_t)
    axs[1].set_title('\n zeros are for times not included in \n the sliding windows')
    fig.tight_layout()
    return t, other_t