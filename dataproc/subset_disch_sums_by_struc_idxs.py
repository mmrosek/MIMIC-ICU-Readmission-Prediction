#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:49:33 2018

@author: miller
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

data_path = "/home/miller/Documents/BDH NLP/Data/"

struc_ids = pd.read_csv(data_path + "struc_hadm_ids.csv")
struc_ids.rename(columns = {"0":"HADM_ID"}, inplace=True)
struc_ids.drop("SUBJECT_ID", inplace=True, axis=1)

X, y = load_svmlight_file(data_path + "struc_data.svmlight")
print("struc data loaded")

text_data = pd.read_csv(data_path + "All Old/sorted_not_matched_w_struc/all_data_sorted.csv") # all data sorted now in new folder

text_data_subset_by_struc_ids = struc_ids.merge(text_data, on = "HADM_ID", how="inner") 

text_data_subset_by_struc_ids.to_csv(data_path + "dis_sums_matched_struc.csv")

data = text_data_subset_by_struc_ids.copy()

train_idxs = int(.9*len(data))
val_idxs = int(.95*len(data))

### Split data into train, validation and test sets
train, validation, test = data[:train_idxs], data[train_idxs:val_idxs], data[val_idxs:]

print("Size of train set: " + str(len(train)))
print("Size of val set: " + str(len(validation)))
print("Size of test set: " + str(len(test)))

if np.sum(data.READMISSION != y) !=0
   print("Mismatched labels")

### Sorting by length for batching
for splt, df in zip(["train","val","test"],[train,validation,test]): 

    df.to_csv('%ssorted_sums_matched_struc_%s.csv' % (data_path, splt), index=False)
    






