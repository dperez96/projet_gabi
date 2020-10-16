import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import medfilt


def zero_extract(series):
    """
    function which removes items in a series equals to zero. If a zero is not followed by another zero, it is replaced
    with a linear interpoation of the items before and after
    :param series: list or numpy array
    :return: the series with the zeros removed
    """
    for i in range(len(series) - 1):
        if series[i] == 0 and series[i + 1] != 0:
            series[i] = (series[i - 1] + series[i + 1]) / 2
        elif series[i] == 0 and series[i + 1] == 0:
            series[i] = series[i - 1]
    series[0] = series[1]
    return series


def new_zero_extract(t, ser):
    """
    function which removes items in a series equals to zero. If a zero is not followed by another zero, it is replaced
    with a linear interpoation of the items before and after. The difference with zero_extract is that in this function,
    a time vector is given as argument, and is cropped as well as the signal
    :param t: list or numpy array
    :param ser: list or numpy array
    :return: lists. cropped versions of t and ser
    """
    new_t, new_ser = [], []
    for i in range(len(ser) - 1):
        if ser[i] != 0:
            new_t.append(t[i])
            new_ser.append(ser[i])

        elif ser[i] == 0 and ser[i + 1] != 0 and i != 0 and ser[i-1] != 0:
            new_t.append(t[i])
            new_ser.append((ser[i - 1] + ser[i + 1]) / 2)
    return new_t, new_ser


def all_zero_extract(t_sig, raw_hr, raw_spo2):  
    """
    function which crops the 3 arrays given in the argument (a time vector, an hr signal and a spo2 signal) at times
    when either the hr signal is equal to 0 or the spo2 signal is under 65%
    :param t_sig: list or nparray
    :param raw_hr: list or nparray
    :param raw_spo2: list or nparray
    :return: lists
    """
    new_t, new_hr, new_spo2 = [], [], []
    for i in range(len(t_sig) - 1):
        if raw_hr[i] != 0 and raw_spo2[i] > 65:
            new_t.append(t_sig[i])
            new_hr.append(raw_hr[i])
            new_spo2.append(raw_spo2[i])
        elif raw_hr[i] == 0 and raw_hr[i+1] != 0 and i != 0 and raw_hr[i-1] != 0:
            new_t.append(t_sig[i])
            new_hr.append((raw_hr[i-1] + raw_hr[i+1])/2)
            new_spo2.append(raw_spo2[i])
    return new_t, new_hr, new_spo2


def convolve_with_dog(y, size):
    """
    apply difference of gaussian filter to a signal.
    :param y: signal
    :param size: size of the window of the convolution
    :return: np array. smoothed signal
    """
    box_pts = size
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 240

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))

    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def SGfilter(y, size):  # apply a Savitzky-Golay filter to the signal
    return savgol_filter(y,  window_length=size, polyorder=3)


def median_filter(y, size):  # apply a median filter to the signal
    filtered_y = medfilt(y, size)
    return filtered_y


def envelope(s):
    """
    function which returns the upper and lower envelope of a signal
    :param s: nparray
    :return: nparray
    """
    t = range(len(s))
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point
    # for both the upper and lower envelope models.

    u_x = [0, ]
    u_y = [s[0], ]

    l_x = [0, ]
    l_y = [s[0], ]

    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (np.sign(s[k] - s[k - 1]) == 1 and np.sign(s[k] - s[k + 1]) == 1) \
                or (np.sign(s[k] - s[k - 1]) == 0 and np.sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (np.sign(s[k] - s[k - 1]) == -1 and (np.sign(s[k] - s[k + 1]) == -1 or np.sign(s[k] - s[k + 1]) == 0))\
                or (np.sign(s[k] - s[k - 1]) == 0 and np.sign(s[k] - s[k + 1]) == -1):
            l_x.append(k)
            l_y.append(s[k])

    # Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for
    # both the upper and lower envelope models.

    u_x.append(len(s) - 1)
    u_y.append(s[-1])

    l_x.append(len(s) - 1)
    l_y.append(s[-1])

    q_u = np.interp(t, u_x, u_y)
    q_l = np.interp(t, l_x, l_y)

    return q_u, q_l


def SpFiltering(s):  # filtering of the spo2 signal : median then average of the envelope
    """
    smoothing function made for the SpO2 signal. the function retunrs the average of the upper and lower envelope of
    the signal
    :param s: list or nparray
    :return: nparray
    """
#    Sp = median_filter(s, win)
    s = np.array(s)
    Sp_u, Sp_l = envelope(s)
    Sp_av = (Sp_u + Sp_l) / 2.0
    return Sp_av

