from matplotlib.mlab import psd
import numpy as np
import matplotlib.pyplot as plt
import spectrum as sp
import filters
import preprocessing as pp


def verif_psd(sig_type):
    file_ids = ['0001', '0193']
    time_windows = [(960, 1260), (299, 599)]
    all_p = []
    all_f = []
    for file_id in file_ids:
        raw_t, raw_hr, t, hr = pp.one_patient_sig(file_id, sig_type)
        for time_window in time_windows:
            print(str(t[time_window[0]]) + '-' + str(t[time_window[1]]))
            p, f = psd(raw_hr[time_window[0]:time_window[1]], NFFT=300, detrend='mean')
            all_p.append(p)
            all_f.append(f)
            plt.figure()
            plt.plot(f, p)
            plt.title(file_id + ' : ' + str(time_window[0]) + '-' + str(time_window[1]))
    return all_f, all_p


raw_t, raw_hr, t, hr = pp.patient_sig('0012')
#t, hr = t[8720:9140], hr[8720:9140]

#ibi = 1/hr

#sig = sp.data_cosine(N=1024, A=0.1, sampling=1024, freq=200) + sp.data_cosine(N=1024, A=0.1, sampling=1024, freq=150)


#pxx1, f1 = psd(raw_hr, NFFT=4096)
#pxx2, f2 = psd(raw_hr, NFFT=4096, detrend='mean')

#w = sp.Window(120, 'hann')
p = sp.WelchPeriodogram(raw_hr[0:300], NFFT=300)
#p.run()

#p.plot()
#plt.plot(p[0][1], 10*np.log10(p[0][0]))
#plt.plot(f1, 10*np.log10(pxx1), label='no detrend')
#plt.plot(f2, 10*np.log10(pxx2), label='mean detrend')

#plt.legend()

