from xml.dom import minidom
import numpy as np
import time
import matplotlib.pyplot as plt
from Constants import *


def events_list(file_id): 
    """
    Each recording is associated with an xml file listing the events ocurring during the recording.
    This function takes reads the xml file for a recording -- whose id is given as argument -
    and returns a list of its events
    :param file_id: id of the recording : 4 digits string
    :return: xml.dom.minicompat.Nodelist. List of the events
    """
    xml_document = minidom.parse('mesa_data/polysomnography/annotations-events-nsrr/mesa-sleep-' + file_id
                                 + '-nsrr.xml')
    return xml_document.getElementsByTagName('ScoredEvent')


def apneas_list(list_of_events, ev_type):
    """
    takes a list of events and returns only the events corresponding to a specific type
    :param list_of_events: xml.dom.minicompat.NodeList (like the output of events_list)
    :param ev_type: string. For example : 'Obstructive apnea|Obstructive Apnea'
    :return: xml.dom.minicompat.NodeList
    """
    list_of_apneas = minidom.NodeList()
    for event in list_of_events:
        if event.childNodes[3].childNodes[0].nodeValue in ev_type:
            list_of_apneas.append(event)
    return list_of_apneas


def times_of_event(t, list_of_event):
    """
    takes the list of events, and a time vector, and returns a np array equal to 1 at the indexes corresponding to times
    when an event of the list is occurring, and zero elsewhere.
    :param t: list or nparray
    :param list_of_event: xml.dom.minicompat.NodeList
    :return: np.ndarray
    """
    ap_times = np.zeros(len(t))
    for event in list_of_event:
        start = round(float(event.childNodes[5].childNodes[0].nodeValue))
        duration = round(float(event.childNodes[7].childNodes[0].nodeValue))
        ap_times[start:start+duration] = 1
    return ap_times


def times_of_event2(list_of_event):
    """
    takes the list of events and returns a list of tuples with the time of beginning and end of each event
    :param list_of_event: xml.dom.minicompat.NodeList
    :return: list of tuples
    """
    timestamps = []
    for event in list_of_event:
        start = round(float(event.childNodes[5].childNodes[0].nodeValue))
        duration = round(float(event.childNodes[7].childNodes[0].nodeValue))
        stop = start+duration
        timestamps.append((start, stop))
    return timestamps


def average_length(file_id, ev_type):
    """
    computes the average duration of a specified type of event for 1 recording
    :param file_id: 4 digits string
    :param ev_type: string. for example 'Obstructive apnea|Obstructive Apnea'
    :return: float
    """
    events = events_list(file_id)
    apneas = apneas_list(events, ev_type)
    times = np.zeros(len(apneas))
    if len(apneas) != 0:
        for i in range(len(apneas)):
            times[i] = float(apneas[i].childNodes[7].childNodes[0].nodeValue)
        return times.mean()
    else:
        return -1


def all_averages(ev_type):
    """
    computes the average durations for every individual
    """
    all_av = []
    for file_id in FILE_IDs:
        av = average_length(file_id, ev_type)
        if av > 0:
            all_av.append((file_id, av))
    return all_av


def prepare_events_data(file_id, event_type):
    """
    for 1 recording, returns a list of tuples with the timestamps of the events of type event_type
    :param file_id: 4 digits string
    :param event_type: string. for example 'Obstructive apnea|Obstructive Apnea'
    :return: list of tuples
    """
    all_events = events_list(file_id)
    events = apneas_list(all_events, event_type)
    times = times_of_event2(events)
    return times


def contains_event(event_times, epoch):
    """
    takes in input a list of event times, and an epoch, and returns 1 if the epoch overlaps with one of the events,
    and 0 otherwise. More than half of the duration of the event should be in the epoch.
    :param event_times: list of tuples
    :param epoch: list or tuple whose first item is the time of beginning and the second item is the time of end of the
    epoch
    :return: boolean (1 if the epochs contains one of the events in the list)
    """
    res = 0
    for temps in event_times:
        dur_tot = temps[1] - temps[0]
        if temps[1] > epoch[0] and temps[0] < epoch[1] and dur_tot > 0:
            # if the event starts after the beggining of the epoch and
            # ends after the start of the epoch  --> condition to have an overlap between the event and the epoch
            if temps[0] < epoch[0]:  # computation of the length of the overlap
                dur = temps[1] - epoch[0]
            elif temps[1] > epoch[1]:
                dur = epoch[1] - temps[0]
            else:
                dur = dur_tot
            if float(dur)/float(dur_tot) > 0.5:  # an event is considered included in the epoch where more than half
                # of the event is occurring
                res = 1
    return res


def apnea_histogram():
    """
    computes histogram of the apneas lengths
    :return: tuple. histogram from matplotlib.pyplot
    """
    hist = []
    for file_id in FILE_IDs:
        print(str(FILE_IDs.index(file_id)) + '/' + str(len(FILE_IDs)), end=', ')
        events = events_list(file_id)
        apneas = apneas_list(events, 'Obstructive apnea|Obstructive Apnea')
        times = times_of_event2(apneas)
        for item in times:
            hist.append(item[1] - item[0])
    return plt.hist(hist, bins='auto')


def remove_wake(t, sigs, file_id):
    """
    removes the time in a list of signals when the individual is awake.
    if there is only 1 signal given in argument, it should be in an list
    :param t: list or nparray. Time vector
    :param sigs: list of lists or nparrays. Signals to crop
    :param file_id: 4 digits string
    :return: the cropped time vector and list of signal(s). the signals are returned as nparrays
    """
    events = events_list(file_id)
    wake_list = apneas_list(events, 'Wake|0')
    wake_times = times_of_event2(wake_list)  # computation of the list of timestamps of beginning and end of each waking
    # period
    t = np.array(t)

    for wake_time in wake_times:
        if wake_time[0] == 0:  # there was an error when the start of the first waking period was 0
            wake_time = (wake_time[0] + 1, wake_time[1])
        wake_start = np.where(t == np.float(wake_time[0]))[0]
        # computation of the index in t of the times of beginning and end of the waking time

        wake_stop = np.where(t == int(wake_time[1]))[0]
        if wake_stop - wake_start > 1800:
            # threshold for the minimum duration of the period where we want to crop it from the signal
            t = np.delete(t, np.arange(wake_start, wake_stop))
            for i in range(len(sigs)):
                sigs[i] = np.delete(sigs[i], np.arange(wake_start, wake_stop))
    return list(t), sigs


def print_db_timestamps(file_id, event_type):
    """
    prints the timestamps of events whose types are in the list given as argument
    :param file_id: 4 digits string
    :param event_type: string, for example 'Obstructive apnea|Obstructive Apnea'
    :return: list of tuples
    """
    all_events = events_list(file_id)
    db_list = apneas_list(all_events, event_type)
    new_db_list = []
    for event in db_list:
        ev_name = event.childNodes[3].childNodes[0].nodeValue
        ev_start = int(float(event.childNodes[5].childNodes[0].nodeValue))
        ev_start_str = time.strftime('%H:%M:%S', time.gmtime(ev_start))
        ev_stop = int(float(ev_start + float(event.childNodes[7].childNodes[0].nodeValue)))
        ev_stop_str = time.strftime('%H:%M:%S', time.gmtime(ev_stop))
        new_db_list.append((ev_start, ev_stop))
        print(ev_name + ' : from ' + ev_start_str + ' to ' + ev_stop_str)
    return new_db_list
