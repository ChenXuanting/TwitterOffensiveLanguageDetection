import os
import pickle
import numpy as np
from utils import eval_stats
from DataLoader import load_test_data

path = f'cached-results/'

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.p' in file:
            files.append(os.path.join(r, file))
print("Total number of predictions:",len(files))
for f in files:
    print(f)

"""Emsemble preds from different models"""
preds = []
for f in files:
    preds.append(pickle.load(open( f, "rb" )))


ensemble_preds = np.round(np.array(preds).mean(axis = 0))

X_test = load_test_data("testing_only", 128)
result = eval_stats(np.array(X_test['label'].values.tolist()), ensemble_preds)

print("NOT:" + "\t" +  "P: %s" %(str(round(result["p_not"]*100, 3))) + "\t" +  "R: %s" %(str(round(result["r_not"]*100, 2))) + "\t" +  "F1: %s" %(str(round(result["f1_not"]*100, 2))))
print("OFF:" + "\t" +  "P: %s" %(str(round(result["p_off"]*100, 2))) + "\t" +  "R: %s" %(str(round(result["r_off"]*100, 2))) + "\t" +  "F1: %s" %(str(round(result["f1_off"]*100, 2))))
print("F1: %s" %(str(round(result["f1"]*100, 2))) + "\t" + "ACC: %s" %(str(round(result["acc"]*100, 2))))