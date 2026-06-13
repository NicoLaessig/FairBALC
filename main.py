import warnings
import argparse
import ast
import copy
import itertools
import subprocess
import json
import shelve
import time
import joblib
import re
import random
import math
import traceback
import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from aif360.datasets import BinaryLabelDataset, StructuredDataset
from scipy.stats import pearsonr
import algorithm
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.5, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--favored", type=str, help="Tuple of values of privileged group.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--metric", default="mean", type=str, help="Metric which will be used to test\
    the classifier combinations. Default metric: mean.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
parser.add_argument("--models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
    tuning should be performed. Else, default parameter values are used. Default: False")
parser.add_argument("--lam", default=0.5, type=float, help="Value of the fairness weight.")
parser.add_argument("--local_lam", default=0.0, type=float, help="Value of the local fairness weight.")
args = parser.parse_args()

input_file = args.ds
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
favored = ast.literal_eval(args.favored)
label = args.label
metric = args.metric
randomstate = args.randomstate
if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)
model_list = ast.literal_eval(args.models)
tuning = args.tuning == "True"
lam = float(args.lam)
local_lam = float(args.local_lam)
model_to_clf = dict()
reg, rem_prot = False, False

df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
error_df = pd.read_csv("configs/ERRORS.csv")

X = df.loc[:, df.columns != label]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=randomstate)

grouped_df = df.groupby(sens_attrs)
group_keys = grouped_df.groups.keys()

train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
classes = len(df[label].unique())
if classes <= 2:
    dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
    dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
    full_dataset = BinaryLabelDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)
else:
    dataset_train = StructuredDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
    dataset_test = StructuredDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
    full_dataset = StructuredDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)
y_train = y_train.to_frame()
y_test = y_test.to_frame()
result_df = copy.deepcopy(y_test)

privileged_groups = []
unprivileged_groups = []
priv_dict = dict()
unpriv_dict = dict()

if isinstance(favored, tuple):
    for i, fav_val in enumerate(favored):
        priv_dict[sens_attrs[i]] = fav_val
        all_val = list(df.groupby(sens_attrs[i]).groups.keys())
        for poss_val in all_val:
            if poss_val != fav_val:
                unpriv_dict[sens_attrs[i]] = poss_val
else:
    if favored == 0:
        priv_dict[sens_attrs[0]] = 0
        unpriv_dict[sens_attrs[0]] = 1
    elif favored == 1:
        priv_dict[sens_attrs[0]] = 1
        unpriv_dict[sens_attrs[0]] = 0

privileged_groups = [priv_dict]
unprivileged_groups = [unpriv_dict]


for sens in sens_attrs:
    result_df[sens] = X_test[sens]
df_dict = dict()
df_dict["filename"] = input_file
df_dict["sens_attrs"] = sens_attrs
df_dict["favored"] = favored
df_dict["label"] = label
df_dict["privileged_groups"] = privileged_groups
df_dict["unprivileged_groups"] = unprivileged_groups
df_dict["index"] = index

log_regr = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
dectree = DecisionTreeClassifier(max_depth=1)

params = json.load(open('configs/params.json'))

for model in model_list:
    print(model)
    try:
        if tuning:
            paramlist = list(params[model]["tuning"].keys())
            parameters = []
            if not opt:
                for param in paramlist:
                    parameters.append(params[model]["tuning"][param])
            else:
                for param in paramlist:
                    parameters.append([opt_param[model][input_file][str(randomstate)][param]])
            full_list = list(itertools.product(*parameters))
            do_eval = True
        else:
            paramlist = list(params[model]["default"].keys())
            li = []
            for param in paramlist:
                li.append(params[model]["default"][param])
            full_list = [li]
            do_eval = False

        real = [item for sublist in dataset_test.labels.tolist() for item in sublist]
        max_val = 0
        best_li = 0

        for i, li in enumerate(full_list):
            iteration_start = time.time()
            score = 0
            try:
                if model == "LFR":
                    clf = algorithm.AIF_LFR(df_dict, log_regr, k=li[3], Ax=li[0], Ay=li[1], Az=li[2], transform=li[3]=="True", remove=rem_prot)
                elif model == "Reweighing":
                    clf = algorithm.AIF_Reweighing(df_dict, log_regr, remove=rem_prot)
                elif model == "EqOddsPostprocessing":
                    clf = algorithm.AIF_EqOddsPostprocessing(df_dict, log_regr, remove=rem_prot, training=True)
                elif model == "RejectOptionClassification":
                    clf = algorithm.AIF_RejectOptionClassification(df_dict, log_regr, metric, eps=li[0], remove=rem_prot, training=True)
                elif model == "AdaFair":
                    clf = algorithm.AdaFairClass(df_dict, dectree, metric, estimators=li[0], c=li[1], CSB=li[2])
                elif model == "FAGTB":
                    clf = algorithm.FAGTBClass(df_dict, estimators=li[0], learning_rate=li[1], lam=li[2], remove=rem_prot)
                elif model == "FairBoost":
                    clf = algorithm.FairBoost(df_dict, log_regr, estimators=li[0], eps=li[1], k=li[2])
                elif model == "FALCC":
                    clf = algorithm.FALCC(link, df, input_file, df_dict, metric, lam=li[0], training=li[1], proxy=li[2], sbt=li[3]=="True", ccr=li[4], reg=reg, iteration=i)
                elif model == "FairBALC":
                    clf = algorithm.FairBALC(link, df, input_file, df_dict, metric, training=li[0], proxy=li[1], ccr=li[2], bal=li[3], eval_strategy=li[4], lam=li[5], ensembling_strategy=li[6], iteration=i)
                elif model == "LogisticRegression":
                    clf = algorithm.LogisticRegressionClass(df_dict, remove=li[0]=="True")

                X_train2 = copy.deepcopy(X_train)
                X_test2 = copy.deepcopy(X_test)
                y_train2 = copy.deepcopy(y_train)
                y_test2 = copy.deepcopy(y_test)
                
                clf.fit(X_train2, y_train2)
                pred = clf.predict(X_test2)

                try:
                    # store the clf in a dictionary
                    clf_copy = copy.deepcopy(clf)
                    model_to_clf[model] = clf_copy
                except:
                    pass

            except Exception as e:
                print("------------------")
                print(model)
                print(e)
                print(traceback.format_exc())
                print("------------------")
            
            if pred is not None:
                modelname = f"{model}_{i}" if tuning else model
                result_df[modelname] = pred
                result_df.to_csv(f"{link}{modelname}_prediction.csv", index_label="index")
                result_df = result_df.drop(columns=[modelname])

    except Exception as E:
        print("---------")
        print(E)
        print("---------")
        err_count = len(error_df)
        error_df.at[err_count, "dataset"] = input_file
        error_df.at[err_count, "model"] = model
        error_df.at[err_count, "error_type"] = str(type(E))
        error_df.at[err_count, "error_msg"] = str(E)

error_df.to_csv("configs/ERRORS.csv", index=False)
