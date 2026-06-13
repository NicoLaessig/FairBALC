"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import re
import copy
import random
import pandas as pd

#For the experiment with two protected attributes, the dataset dictionary has to add the second attribute
#to the sens_attrs list and the corresponding value of the privileged group to the favored tuple.
DATA_DICT = {
    "compas_race": {"sens_attrs": ["Race"], "label": "Two_yr_Recidivism", "favored": (0)},
    "compas_sex": {"sens_attrs": ["Female"], "label": "Two_yr_Recidivism", "favored": (0)},
    "communities": {"sens_attrs": ["race"], "label": "crime", "favored": (0)},
    "credit_card_clients": {"sens_attrs": ["sex"], "label": "payment", "favored": (1)},
    "adult_data_set_race": {"sens_attrs": ["race"], "label": "salary", "favored": (0)},
    "adult_data_set_sex": {"sens_attrs": ["sex"], "label": "salary", "favored": (0)},
    "ACS_ID_norm": {"sens_attrs": ["SEX"], "label": "PINCP", "favored": (1)},
    "ACS_OR_norm": {"sens_attrs": ["SEX"], "label": "PINCP", "favored": (1)},
    "drug_consumption": {"sens_attrs": ["gender"], "label": "cannabis", "favored": (1)},
    }

testsize = 0.3
tuning = False
lam = 0.5

models = [
    "FairBALC",
    "FALCC",
    "AdaFair",
    "FairBoost",
    "FAGTB",
    "Reweighing",
    "LFR",
    "RejectOptionClassification",
    "EqOddsPostprocessing"
]

models_eval = []
for model in models:
    for i in range(200):
        models_eval.append(f"{model}_{i}")

ds = "communities"
metric = "demographic_parity"
randomstate = random.randint(0,1000)

sensitive = DATA_DICT[ds]["sens_attrs"]
label = DATA_DICT[ds]["label"]
favored = DATA_DICT[ds]["favored"]

link = "Results/" + str(metric) + "/" + str(ds) + "/" + str(randomstate) + "/"

try:
    os.makedirs(link)
except FileExistsError:
    pass

proc = subprocess.check_call(['python', '-Wignore', 'main.py', '--output', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--testsize', str(testsize), '--randomstate', str(randomstate),
    '--models', str(models), '--metric', str(metric), '--tuning', str(tuning)])

proc = subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--folder', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--models', str(models_eval), '--metric', str(metric),
    '--name', 'EVALUATION', '--lrd', str(False)])
