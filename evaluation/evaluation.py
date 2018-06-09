"""
    This file contains evaluation methods that take in a set of predicted labels
        and a set of ground truth labels and calculate precision, recall, accuracy, and f1 score
"""
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from constants import *

def all_metrics(yhat_raw, y, thresh = None):
    """
        Inputs:
            yhat: binarized predictions
            y: binary ground truth matrix
            yhat_raw: raw predictions (floats)
        Outputs:
            dict holding relevant metrics
    """
    #micro
    ymic = y.ravel() # RAVEL = FLATTEN TO 1D
    yhat_raw = yhat_raw.ravel()

    if thresh: # If binarization threshold is passed (for test set)

        names = ["acc", "prec", "rec", "f1", "passed_f1_thresh"]
        micro = all_micro(yhat_raw, ymic, thresh)
        metrics = {names[i] + "_micro": micro[i] for i in range(len(micro))}
        roc_auc = auc_metrics(yhat_raw, ymic, metrics['passed_f1_thresh_micro'])

    else:

        names = ["acc", "prec", "rec", "f1", "opt_f1_thresh"]
        micro = all_micro(yhat_raw, ymic)
        metrics = {names[i] + "_micro": micro[i] for i in range(len(micro))}
        roc_auc = auc_metrics(yhat_raw, ymic, metrics['opt_f1_thresh_micro'])

    metrics.update(roc_auc)

    return metrics

def all_micro(yhat_raw, ymic, thresh = None):

    if thresh:
        y_bin = np.where(yhat_raw > thresh, 1, 0)
        f1 = micro_f1(y_bin, ymic)
        opt_thresh = thresh

    else:
        f1, opt_thresh = find_opt_thresh_f1(yhat_raw, ymic, 0.01, 1, 100)

    yhatmic = np.where(yhat_raw > opt_thresh, 1, 0)
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), f1, opt_thresh

def micro_accuracy(yhatmic, ymic):
    return np.sum(yhatmic == ymic) / ymic.shape[0]

def micro_precision(yhatmic, ymic):
    if np.sum(yhatmic) == 0:
        return 0
    else:
        return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def find_opt_thresh_f1(yhat_raw, y, min_thresh, max_thresh, num_thresh):

    max_f1 = 0
    max_thresh_ = 0
    thresh_list = list(np.linspace(min_thresh,max_thresh,num_thresh))

    for thresh in thresh_list:
        yhat = np.where(yhat_raw > thresh, 1, 0) # Thresholding of preds
        f1 = micro_f1(yhat, y)

        if f1 >= max_f1:
            max_f1 = f1
            max_thresh_ = thresh

    return max_f1, max_thresh_


def auc_metrics(yhat_raw, y, thresh):
    if yhat_raw.shape[0] <= 1:
        return

    roc_auc = {}

    #micro-AUC
    yhat_bin = np.where(yhat_raw > thresh, 1, 0)
    roc_auc['true_neg'], roc_auc['false_pos'], roc_auc['false_neg'], roc_auc['true_pos'] = confusion_matrix(y, yhat_bin).ravel()

    # Converting to float to allow serialization to json
    roc_auc['true_neg'] = np.float64(roc_auc['true_neg'])
    roc_auc['false_neg'] = np.float64(roc_auc['false_neg'])
    roc_auc['true_pos'] = np.float64(roc_auc['true_pos'])
    roc_auc['false_pos'] = np.float64(roc_auc['false_pos'])

    roc_auc["auc"] = roc_auc_score(y, yhat_raw) # (groundTruth, preds)

    return roc_auc

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_metrics(metrics):
    #annoyingly complicated printing, to keep track of progress during training

    if "opt_f1_thresh_micro" in metrics.keys():
        print("\n f-1, opt_thresh, AUC,   acc,   prec, recall,  True P,  False P, True N, False N")
        print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["f1_micro"], metrics["opt_f1_thresh_micro"], metrics["auc"], metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["true_pos"], metrics["false_pos"],metrics["true_neg"], metrics["false_neg"]))

    else:
        print("\nf-1, passed_thresh, AUC,   acc,   prec, recall,  True P,  False P, True N, False N")
        print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["f1_micro"], metrics["passed_f1_thresh_micro"], metrics["auc"], metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["true_pos"], metrics["false_pos"],metrics["true_neg"], metrics["false_neg"]))

