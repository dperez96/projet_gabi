import pandas as pd
QUALITY_VARIABLES = [('quoxim5', 3, '+'), ('overall5', 4, '+'), ('respevpr5', 1, '-')]
"""for each tuple, the first item of QUALITY_VARIABLES is the variable to check, the second is a threshold, and if the 
third item is '+', it means that the recordings select are the ones for which the value of this variable it strictly 
superior to the threshold, and strictly inferior if the third item is '-'

Documentation about all the variables can be found at 
'mesa_data/documentation/dataset-descriptions/MESAe5a113SleepPolysomn_20150630.pdf'
"""


def ahi_list(file_ids):
    """
    returns list of apnea-hypopnea-indexes of the individuals in a list
    """
    data = pd.read_csv('mesa_data/datasets/mesa-sleep-dataset-0.3.0 (1).csv')
    id = data['mesaid'].values
    ahi = data['oahi35'].values
    new_id, new_ahi = [], []
    for i in range(len(id) - 1):
        if id[i] in [int(elem) for elem in file_ids]:
            new_id.append(id[i])
            new_ahi.append(ahi[i])
    return new_id, new_ahi


def check_variable_quality(variable, threshold):
    """
    for a certain type of variable and a certain threshold given in input, the function prints the ids of all the
    individuals for which the variable surpasses the threshold set
    """
    data = pd.read_csv('mesa_data/datasets/mesa-sleep-dataset-0.3.0 (1).csv')  # load the csv with all the variables
    # values
    id = data['mesaid'].values
    variable_quality = data[variable].values
    for i in range(len(variable_quality)):
        if variable_quality[i] > threshold:
            print(str(id[i]) + ' - ' + str(variable_quality[i]))


def recordings_list():
    """
    Functions which returns the list of recordings which comply to the criteria of the variables in 'QUALITY_VARIABLES'
    """
    variables = []  # array which will contain the values of the variables in QUALITY_VARIABLES for each recording
    boolean_list = []  # array of the same shape as variables, but for each item, instead of the the value of the
    # variable, there is a boolean indicating if the value complies the the criteria of QUALITY_VARIABLES
    data = pd.read_csv('mesa_data/datasets/mesa-sleep-dataset-0.3.0 (1).csv')  # load the csv with all the variables
    # values
    ids = data['mesaid'].values
    for item in QUALITY_VARIABLES:
        variables.append(data[item[0]].values)  # data[item[0]].values is a list of the values of the variable item for
        # each recording
    for i in range(len(variables[0])):
        individual_booleans = []  # a list of booleans for one individual, where each boolean corresponds to one fo the
        # variables
        for j in range(len(QUALITY_VARIABLES)):
            threshold = QUALITY_VARIABLES[j][1]
            if QUALITY_VARIABLES[j][2] == '+':
                individual_booleans.append(variables[j][i] > threshold)  # testing for each variable if it complies with
                # the criteria
            else:
                individual_booleans.append(variables[j][i] < threshold)
        boolean_list.append(individual_booleans)
    selected_ids = []
    for i in range(len(ids)):
        if boolean_list[i][0] and boolean_list[i][1] and boolean_list[i][2]:  # for a recording to be selected, all its
            # booleans should be 'True' (each variable comply to the criteria of QUALITY_VARIABELS)
            selected_ids.append(ids[i])
    return selected_ids
