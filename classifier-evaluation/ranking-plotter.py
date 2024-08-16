# Code to plot ranking results of four classifiers on two data sets
#
# Code by Sumayya Ziyad and Peter Christen.
#
# Usage: python3 rank-plotter.py  [data_set]  [bw_color]
#
# where:
# - data_set  is either 'b' for Breast cancer data set or 'g' for German
#             credit data set.
# bw_color    is either 't' for generating color plots or 'f' for generating
#             black-white plots.


import sys

import pandas as pd
import matplotlib.pyplot as plt

ds = sys.argv[1].lower()[0]
assert ds in ['b','g']

color = sys.argv[2].lower()[0]
assert color in ['t', 'f']
if (color == 't'):
  color = True
else:
  color = False

# The list of how performance measures should be ordered in the plot
#
perf_meas_plot_list = ['ACC', 'BACC', 'GACC', 'PR/PPV', 'REC/SE', 'F1', 'MCC', 'FM',
                       'J/INF', 'MAR']

num_pref_meas = len(perf_meas_plot_list)

plt.rcParams.update({'font.size': 14})

breast_cancer_data = {
    'Performance Measure': ["ACC", "BACC", "F1", "FM", "GACC", "J/INF", "MAR", "MCC", "PR/PPV", "REC/SE"],
    'Decision Tree':       [4, 3, 4, 4, 1, 3, 4, 4, 3, 4],
    'Logistic Regression': [3, 2, 3, 3, 3, 2, 2, 2, 2, 2],
    'Random Forest':       [2, 1, 2, 2, 2, 1, 1, 1, 1, 3],
    'SVM':                 [1, 4, 1, 1, 4, 4, 3, 3, 4, 1]
}

german_credit_data = {
    'Performance Measure': ["ACC", "BACC", "F1", "FM", "GACC", "J/INF", "MAR", "MCC", "PR/PPV", "REC/SE"],
    'Decision Tree':       [4, 4, 4, 4, 3, 4, 4, 4, 3, 4],
    'Logistic Regression': [2, 1, 3, 3, 1, 1, 3, 2, 1, 3],
    'Random Forest':       [1, 2, 1, 2, 2, 2, 1, 1, 2, 2],
    'SVM':                 [3, 3, 2, 1, 4, 3, 2, 3, 4, 1]
}

# Generate the actual data for plotting
#
plot_data = {'Performance Measure':perf_meas_plot_list}
plot_data['Decision Tree'] =       []
plot_data['Logistic Regression'] = []
plot_data['Random Forest'] =       []
plot_data['SVM'] =                 []
    
for perf_meas in perf_meas_plot_list:
  if (ds == 'b'):
    perf_meas_ind = breast_cancer_data['Performance Measure'].index(perf_meas)
  else:
    perf_meas_ind = german_credit_data['Performance Measure'].index(perf_meas)
  print(perf_meas, perf_meas_ind)

  for class_meth in ['Decision Tree', 'Logistic Regression',
                     'Random Forest', 'SVM']:
    if (ds == 'b'):
      meas_rank = breast_cancer_data[class_meth][perf_meas_ind]
    else:
      meas_rank = german_credit_data[class_meth][perf_meas_ind]
    plot_data[class_meth].append(meas_rank)

for (k,v) in plot_data.items():
  print(k,v)

#
#  df = pd.DataFrame(breast_cancer_data)
#  num_class = len(breast_cancer_data.keys()) - 1
#else:
#  df = pd.DataFrame(german_credit_data)
#  num_class = len(german_credit_data.keys()) - 1

df =        pd.DataFrame(plot_data)
num_class = len(plot_data.keys()) - 1

# Set the index column
#
df.set_index('Performance Measure', inplace=True)

class_init_list = []

fig, ax = plt.subplots(figsize=(num_pref_meas+2, num_class+2))
for column in df.columns:
    class_init_list.append(column[0])
    if (color == False):
      ax.plot(df.index, df[column], marker='o', markersize=17, label=column, color='k')
    else:
      ax.plot(df.index, df[column], marker='o', markersize=17, label=column)

# to add the ranks on the markers; (use zorder to raise the digits vertically)
for i in range(df.shape[0]):
    for j in range(0, df.shape[1]):
#        ax.text(df.index[i], df.iloc[i, j], str(df.iloc[i, j]), ha='center', va='center', color="white")
# Peter's version, first letter of classifier method in nodes
                ax.text(df.index[i], df.iloc[i, j], class_init_list[j], ha='center', va='center', color="white")

plt.ylim(0.7, 4.3)

# Reverse the y-axis to have rank 1 at the top
ax.invert_yaxis()

ax.set_yticks(range(1, num_class + 1))

if (ds == 'b'):
  ax.set_title('Ranking of classifiers for the "Breast Cancer" data set')
else:
  ax.set_title('Ranking of classifiers for the "German Credit" data set')

# Add labels and title
ax.set_xlabel('Performance Measure')
ax.set_ylabel('Rank')

plt.tight_layout()

if (ds == 'b'):
  if (color == False):
    #plt.savefig("bump_chart_breast_cancer_classifier_rankings_with_marker_text-bw.eps", format="eps")
    #plt.savefig("bump_chart_breast_cancer_classifier_rankings_with_marker_text-bw.pdf", format="pdf")
    plt.savefig("bump_chart_breast_cancer_classifier_rankings_with_marker_text-bw.svg", format="svg")
  else:
    #plt.savefig("bump_chart_breast_cancer_classifier_rankings_with_marker_text.eps", format="eps")
    #plt.savefig("bump_chart_breast_cancer_classifier_rankings_with_marker_text.pdf", format="pdf")
    plt.savefig("bump_chart_breast_cancer_classifier_rankings_with_marker_text.svg", format="svg")

else:
  if (color == False):
    #plt.savefig("bump_chart_german_credit_classifier_rankings_with_marker_text-bw.eps", format="eps")
    #plt.savefig("bump_chart_german_credit_classifier_rankings_with_marker_text-bw.pdf", format="pdf")
    plt.savefig("bump_chart_german_credit_classifier_rankings_with_marker_text-bw.svg", format="svg")
  else:
    #plt.savefig("bump_chart_german_credit_classifier_rankings_with_marker_text.eps", format="eps")
    #plt.savefig("bump_chart_german_credit_classifier_rankings_with_marker_text.pdf", format="pdf")
    plt.savefig("bump_chart_german_credit_classifier_rankings_with_marker_text.svg", format="svg")

plt.show()
