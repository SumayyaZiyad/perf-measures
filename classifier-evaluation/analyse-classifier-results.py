# Code to analyse the classifier evaluation results obtained through classifier-eval.py
#
# Code by Sumayya Ziyad and Peter Christen - April 2024
#
# Usage: python3 analyse-classifier-results.py [path_to_results_file]

import csv
import sys


name_abbrv_map_dict = {'Accuracy': 'ACC', 'Balanced accuracy': 'BAL',
                       'Precision': 'PRE', 'Recall': 'REC', 'F-measure': 'FME',
                       'Fowlkes-Mallows index': 'FMI',
                       'Matthews corr coef': 'MCC', 'Geo mean': 'GME',
                       'Informedness': 'INF', 'Markedness': 'MAR'}

f = open(sys.argv[1], 'rt')

csv_reader = csv.reader(f)

header_list = next(csv_reader)
print('Header line:', header_list)
num_attr = len(header_list)

perf_meas_list = header_list[6:]
num_perf_meas = len(perf_meas_list)

data_set_res_dict = {}

for row_list in csv_reader:
    print(row_list)

    data_set = row_list[0]
    classifier = row_list[1]

    # For each performance measure (key) a list of results for the different
    # classifiers
    #
    perf_meas_dict = data_set_res_dict.get(data_set, {})

    for (i, perf_meas) in enumerate(perf_meas_list):
        res_val = float(row_list[i + 6])

        pref_meas_res_list = perf_meas_dict.get(perf_meas, [])
        pref_meas_res_list.append((res_val, classifier))

        perf_meas_dict[perf_meas] = pref_meas_res_list

    data_set_res_dict[data_set] = perf_meas_dict

for data_set in sorted(data_set_res_dict.keys()):
    print()
    perf_meas_dict = data_set_res_dict[data_set]
    print('Ranking of classifiers for different performance measures for  data set:', data_set)
    print()
    for perf_meas in sorted(perf_meas_dict.keys()):
        print('  %s:' % perf_meas)
        pref_meas_res_list = sorted(perf_meas_dict[perf_meas])
        for (res_val, classifier) in pref_meas_res_list:
            print('    %20s: %.4f' % (classifier, res_val))
        print()

    print('- ' * 40)

sys.exit()
