import matplotlib.pyplot as plt
import fct_events as fe
import features_service as fs
from matplotlib.mlab import psd
from time import time
from Constants import *
import time_service as ts
import preprocessing as pp
import scipy


def plot_events(t, signal, list_of_events):
    apnea_time = fe.times_of_event(t, list_of_events[2:4])
    timeStamps = fe.times_of_event2(list_of_events[2:4])
    for event in timeStamps:
        plt.figure()
        start = event[0]
        stop = event[1]
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(t[start-30:stop+30], signal[start-30:stop+30])
        axs[1].plot(t[start-30:stop+30], apnea_time[start-30:stop+30])


def plot_psd(t_sig, raw_sig, sig, start, stop):


    pxx, freqs = psd(sig[start:stop])
    plt.title('smoothed hr psd')
    plt.figure()
    raw_pxx, raw_freqs = psd(raw_sig[start:stop])
    plt.title('raw hr psd')

    plt.figure()
    plt.plot(t_sig[start:stop], sig[start:stop])

    return pxx, freqs, raw_pxx, raw_freqs


def boxplot_lfe(lfe_a, lfe_wa, f_min, f_max, ep_len):
    val_a = [val[0] for val in lfe_a]
    val_wa = [val[0] for val in lfe_wa]

    plt.figure()
    box_data = plt.boxplot([val_a, val_wa])
    plt.grid()
    plt.title('freq: ' + str(f_min) + ' - ' + str(f_max) + '\nEpoch: ' + str(ep_len))
    return box_data


def plot_psd2(list_of_psd, timestamp):  # timestamp must be multiple of 60 seconds
    index = timestamp/60 + 1
    plt.figure()
    plt.plot(list_of_psd[index][0], list_of_psd[67][1])
    plt.title(str(list_of_psd[67][2][0]) + '-' + str(list_of_psd[67][2][1]))


def plot_3_features(features):
    # computes and represents in a 3D scatter-plot the 3 features given in list as argument
    start = time()  # a = apnea, wa = without apnea
    features_a, features_wa = fs.make_feature_all(features)
    # the elements of feature are tuples (signal_type, feature_type)
    print('tot : ' + str(time() - start))

    feat_1_a = [elem[0] for elem in features_a]
    # putting the first feature computed on each epoch with apnea of each individual in a list
    feat_1_wa = [elem[0] for elem in features_wa]

    feat_2_a = [elem[1] for elem in features_a]
    feat_2_wa = [elem[1] for elem in features_wa]

    feat_3_a = [elem[2] for elem in features_a]
    feat_3_wa = [elem[2] for elem in features_wa]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    no_ap = plt.scatter(feat_1_wa, feat_2_wa, feat_3_wa, marker='.')
    ap = plt.scatter(feat_1_a, feat_2_a, feat_3_a, marker='.')
    ax.set_xlabel(str(features[0][0]) + '-' + str(features[0][1]))
    ax.set_ylabel(str(features[1][0]) + '-' + str(features[1][1]))
    ax.set_zlabel(str(features[2][0]) + '-' + str(features[2][1]))

    plt.legend((ap, no_ap), ('apnea', 'no_apnea'))
    plt.title('VLF : ' + str(F1_MIN) + '-' + str(F1_MAX) + '\nLF : ' + str(F2_MIN) + '-' + str(F2_MAX))
    plt.show()
    return features_a, features_wa


def plot_vlf_lf(sig_type):
    # computes and plots the vlf and lf of either hr of spo2
    start = time()
    lfe_a, lfe_wa = fs.old_make_feature_all(fs.low_freq_energy, sig_type)  # a = apnea, wa = without apnea
    print('tot : ' + str(time() - start))

    val_a = [elem[0] for elem in lfe_a]
    # putting the first feature computed on each epoch with apnea of each individual in a list
    val_wa = [elem[0] for elem in lfe_wa]

    vlf_a = [elem[0] for elem in val_a]
    lf_a = [elem[1] for elem in val_a]

    vlf_wa = [elem[0] for elem in val_wa]
    lf_wa = [elem[1] for elem in val_wa]

    plt.figure()
    no_ap = plt.scatter(vlf_wa, lf_wa, marker='.')
    ap = plt.scatter(vlf_a, lf_a, marker='.')
    plt.xlabel('VLF')
    plt.ylabel('LF')

    plt.legend((ap, no_ap), ('apnea', 'no_apnea'))
    plt.title('VLF : ' + str(F1_MIN) + '-' + str(F1_MAX) + '\nLF : ' + str(F2_MIN) + '-' + str(F2_MAX))
    plt.show()
    return vlf_wa, lf_wa, vlf_a, lf_a


def plot_peaks(file_id, sig_type):
    # on each plot, the peaks of psd of epochs with apnea are distinguished from those without apnea
    raw_t, raw_sig, t, sig = pp.one_patient_sig(file_id, sig_type)
    epochs = ts.new_sliding_window_array(t, WINDOW, EPOCH)  # generating the windows
    peaks_a_1, peaks_wa_1, peaks_a_2, peaks_wa_2, peaks_a_3, peaks_wa_3 = [], [], [], [], [], []
    # 1st, 2nd and 3rd peak of each psd
    peaks = [(peaks_a_1, peaks_wa_1), (peaks_a_2, peaks_wa_2), (peaks_a_3, peaks_wa_3)]
    for ep in epochs:
        print(str(epochs.index(ep)) + '/' + str(len(epochs)))
        middle = (ep[1] - ep[0]) / 2
        # the epoch in the middle of the window is what is being classified as apnea or not apnea
        mid_ep_start = middle - EPOCH / 2
        mid_ep_stop = middle + EPOCH / 2
        middle_ep = (ep[0] + mid_ep_start, ep[0] + mid_ep_stop)
        apnea_times = fe.prepare_events_data(file_id, ['Obstructive apnea|Obstructive Apnea', 'Hypopnea|Hypopnea'])
        # computing the times where apnea occur
        contains_apnea = fe.contains_event(apnea_times, middle_ep)  # check if the epoch contains an apnea
        p, f = psd(sig[ep[0]:ep[1]], NFFT=WINDOW, detrend='mean')  # computing the psd
        peaks_index = scipy.signal.find_peaks(p)  # computing the indexes of peaks
        if len(peaks_index[0]) > 2:  # if at least 3 peaks were found
            for i in range(3):  # only the 3 first peaks are looked at
                # print(peaks_index[0][i])
                amplitude = p[peaks_index[0][i]]
                if amplitude > 1:  # neglecting peaks smaller than 1
                    if contains_apnea:  # the peaks are added in the appropriate list
                        peaks[i][0].append((f[peaks_index[0][i]], amplitude))
                        print((f[peaks_index[0][i]], amplitude))
                    else:
                        peaks[i][1].append((f[peaks_index[0][i]], amplitude))
                        print((f[peaks_index[0][i]], amplitude))
    plt.figure()
    f_a_1 = [elem[0] for elem in peaks[0][0]]  # generating a list with the frequency of all 1st peaks
    p_a_1 = [elem[1] for elem in peaks[0][0]]  # generating a list with the amplitude of all 1st peaks

    f_a_2 = [elem[0] for elem in peaks[1][0]]
    p_a_2 = [elem[1] for elem in peaks[1][0]]

    f_a_3 = [elem[0] for elem in peaks[2][0]]
    p_a_3 = [elem[1] for elem in peaks[2][0]]

    f_wa_1 = [elem[0] for elem in peaks[0][1]]
    p_wa_1 = [elem[1] for elem in peaks[0][1]]

    f_wa_2 = [elem[0] for elem in peaks[1][1]]
    p_wa_2 = [elem[1] for elem in peaks[1][1]]

    f_wa_3 = [elem[0] for elem in peaks[2][1]]
    p_wa_3 = [elem[1] for elem in peaks[2][1]]

    plt.figure()
    no_ap_1 = plt.scatter(f_wa_1, p_wa_1, marker='.')
    ap_1 = plt.scatter(f_a_1, p_a_1, marker='.')
    plt.legend((no_ap_1, ap_1), ('epoch without apnea', 'epoch with apnea'))
    plt.title('First spike of psd computed on epochs with and without apneas')

    plt.figure()
    no_ap_2 = plt.scatter(f_wa_2, p_wa_2, marker='.')
    ap_2 = plt.scatter(f_a_2, p_a_2, marker='.')
    plt.legend((no_ap_2, ap_2), ('epoch without apnea', 'epoch with apnea'))
    plt.title('Second spike of psd computed on epochs with and without apneas')

    plt.figure()
    no_ap_3 = plt.scatter(f_wa_3, p_wa_3, marker='.')
    ap_3 = plt.scatter(f_a_3, p_a_3, marker='.')
    plt.legend((no_ap_3, ap_3), ('epoch without apnea', 'epoch with apnea'))
    plt.title('Third spike of psd computed on epochs with and without apneas')


def boxplot_1_feature(feature_a, feature_wa, title):
    # makes a boxplot with 2 boxes : 1 for epochs with apnea, 1 for epochs without

    plt.figure()
    box_data = plt.boxplot([feature_a, feature_wa])
    plt.grid()
    plt.title(title)
    plt.xticks([1, 2], ['apnea', 'no apnea'])
    return box_data

