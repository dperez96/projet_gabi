import fct_mesa_data as fmd
import filters
import numpy as np


def one_patient_sig(file_id, sig_type):
    """
    returns the raw and preprocessed signal of the type given in argument
    :param file_id: string. 4 digits id of the recording
    :param sig_type: string. 'HR', or 'SpO2' for example
    :return: 4 np.ndarray
    """
    raw_data = fmd.load_raw(file_id, sig_type)
    raw_t = raw_data[:, 0]
    raw_sig = raw_data[:, 1]
    t, sig = old_sig_preprocessing(raw_t, raw_sig, sig_type)
    t = np.array(t)
    return raw_t, raw_sig, t, sig


def patient_sig(file_id):
    """
    returns the raw and preprocessed hr and spo2 signals
    when a part of one signal is cropped (because it is equal to 0 for example), it is cropped from both hr and spo2
    :param file_id: string. id of the recording to select
    :return: the raw signals are np.ndarray and the new signals are lists
    """
    raw_hr_sig = fmd.load_raw(file_id, 'HR')
    raw_t_hr = raw_hr_sig[:, 0]
    raw_hr = raw_hr_sig[:, 1]

    raw_spo2_sig = fmd.load_raw(file_id, 'Spo2')
    raw_t_spo2 = raw_spo2_sig[:, 0]
    raw_spo2 = raw_spo2_sig[:, 1]

    if len(raw_t_hr) != len(raw_t_spo2):
        print('t_hr différent de t_spo2!')

    # t_sig, [hr, spo2] = fe.remove_wake(raw_t_hr, [raw_hr, raw_spo2], file_id)  # removal of waking time
    new_t_sig, new_hr, new_spo2 = sig_preprocessing(raw_t_hr, raw_hr, raw_spo2)  # (t_sig, hr, spo2) if removal
    # of waking time uncommented

    return raw_t_hr, raw_hr, raw_spo2, new_t_sig, new_hr, new_spo2


def sig_preprocessing(t_sig, raw_hr, raw_spo2):
    """
    returns smoothed and cropped signals
    when a part of one signal is cropped (because it is equal to 0 for example), it is cropped from both hr and spo2
    :param t_sig: np.ndarray
    :param raw_hr: np.ndarray
    :param raw_spo2: np.ndarray
    :param file_id: string
    :return: lists
    """
    new_t, hr, spo2 = filters.all_zero_extract(t_sig, raw_hr, raw_spo2)  # removal of zeroes
    # filtered_hr = filters.SGfilter(hr, hr_filt)
    # filtered_spo2 = filters.SpFiltering(spo2)
    # uncomment in order to apply smoothing on the sigals
    return new_t, hr, spo2  # spo2 filtered_hr


def old_sig_preprocessing(t_sig, sig, data_type):  # takes a type of signal in argument (like 'heart rate', 'spo2, etc.)
    #                                        and returns this signal filtered for one individual
    t_sig, sig = filters.new_zero_extract(t_sig, sig)
    if data_type == 'HR':
        sig = filters.SGfilter(sig, 9)
    elif data_type == 'Spo2':
        # here, the raw spo2 is conserved as the first item of new_sig in order to compare it with the smoothed
        # spo2. This is no longer the case with the function sig_preprocessing
        new_sig = np.zeros((2, len(t_sig)))
        new_sig[0] = sig
        new_sig[1] = filters.SpFiltering(sig, 9)
        sig = new_sig[1]
    else:
        print('no filter for this type')
    return t_sig, sig
