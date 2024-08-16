# Code to classify two datas sets using 4 different classifiers and to evaluate the classifier outcome based on
# different performance measures. This implementation has been adopted to two datasets:
#       1. Breast cancer dataset - https://archive.ics.uci.edu/dataset/14/breast+cancer
#       2. German credit dataset - https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
#
# Code by Sumayya Ziyad and Peter Christen - April 2024
#
# Usage: python3 classifier-eval.py [path_to_breast_cancer_data] [path_to_german_credit_data]


import pprint
import csv
import math
import time
import pandas as pd
import numpy
import sys

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

today_str = time.strftime("%Y%m%d", time.localtime())
numpy.random.seed(42)  # Ensure repeatability

path_to_breast_cancer_data = sys.argv[1]
path_to_german_credit_data = sys.argv[2]

out_csv_name = 'classifier-results-%s.csv' % today_str

results_summary = {}
ranking_summary = {}

# generate one line per experiment with all results
res_csv_list = []
res_csv_header_list = ['Dataset', 'Classifier', 'TP', 'FP', 'TN', 'FN', \
                       'Precision', 'Recall', 'F-measure', 'Accuracy', \
                       'Balanced accuracy', 'Fowlkes-Mallows index', \
                       'Matthews corr coef', 'Geo mean', 'Informedness', \
                       'Markedness']
res_csv_list.append(res_csv_header_list)

dataset_chars = {
    "breast_cancer": {
        "url": path_to_breast_cancer_data,
        "categorical_features": ["age", "menopause", "tumor-size", "inv-node", "node-caps", "breast", "breast-quad",
                                 "irradiat"],
        "target_col": "class",
        "predicted_positive": ["no-recurrence-events"],
        "headers": ["class", "age", "menopause", "tumor-size", "inv-node", "node-caps", "deg-malig", "breast",
                    "breast-quad", "irradiat"],
        "delimiter": ","
    },
    "german-credit": {
        "url": path_to_german_credit_data,
        "categorical_features": ['A1', 'A3', 'A4', 'A6', 'A7', 'A9', 'A10', 'A12', 'A14', 'A15', 'A17', 'A19', 'A20'],
        "target_col": "class",
        "predicted_positive": [1],
        "headers": ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                    'A16', 'A17', 'A18', 'A19', 'A20', 'class'],
        "delimiter": ","
    }
}

must_add_headers = ["breast_cancer", "german-credit"]

for key, value in sorted(dataset_chars.items()):  # Sort to ensure consistency
    if key in must_add_headers:
        df = pd.read_csv(value["url"], header=None, delimiter=value["delimiter"])
    else:
        df = pd.read_csv(value["url"], delimiter=value["delimiter"])

    print(len(df))

    if key in must_add_headers:
        df.columns = value["headers"]
    target_col = value["target_col"]

    X = df.drop(columns=target_col)  # Adjust the target column index as needed
    map_positive_predictions = {pred: 1 for pred in value["predicted_positive"]}
    y = df[target_col].map(map_positive_predictions).fillna(0)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    categorical_cols = value["categorical_features"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [col for col in X.columns if col not in categorical_cols]),
            ('cat', categorical_transformer, categorical_cols)
        ])

    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='auto'),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    pipelines = {}
    for name, classifier in sorted(classifiers.items()):
        pipelines[name] = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

    print(f"Dataset: {key}")

    data_set_summary = {
        "precision": 0,
        "recall": 0,
        "f_score": 0,
        "auc_pr": 0
    }

    for name, pipeline in sorted(pipelines.items()):
        print(f"Classifier: {name}")
        y_pred = cross_val_predict(pipeline, X, y, cv=10)

        # Get confusion matrix counts
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        prec = float(tp) / float(tp + fp)
        reca = float(tp) / float(tp + fn)
        fmea = 2.0 * tp / (2.0 * tp + fp + fn)
        print('    P, R, F1: ', prec, reca, fmea)
        print()

        # Calculate measures from confusion matrix
        acc = float(tp + tn) / float(tp + fp + tn + fn)

        tpr = reca
        tnr = float(tn) / float(tn + fp)
        fpr = float(fp) / float(tn + fp)
        fnr = float(fn) / float(tp + fn)

        neg_pred_val = float(tn) / float(tn + fn)
        false_disc_rate = float(fp) / float(tp + fp)
        false_omi_rate = float(fn) / float(tn + fn)

        bal_acc = (tpr + tnr) / 2.0
        fmi = math.sqrt(prec * reca)  # Fowlkes-Mallows index

        geo_mean = math.sqrt(tpr * tnr)  # Geometric mean

        inf = tpr - fpr  # Informedness
        mar = prec + neg_pred_val - 1  # Markedness

        # Matthew correlation coefficient (-1 to 1)
        mcc = float(tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        nlr = fnr / tnr  # negative likelihood ratio (not 0..1)
        plr = tpr / fpr  # positive likelihood ratio (not 0..1)

        res_list = [key, name, tp, fp, tn, fn, prec, reca, fmea, acc, bal_acc, fmi, mcc, geo_mean, inf, mar]
        res_csv_list.append(res_list)
        assert len(res_list) == len(res_csv_list[0])

        print(res_list)

pprint.pprint(ranking_summary)
pprint.pprint(results_summary)

out_file = open("results/" + out_csv_name, 'wt')
csv_writer = csv.writer(out_file)

print()
for res_list in res_csv_list:
    print(res_list)
    csv_writer.writerow(res_list)

out_file.close()
print()
