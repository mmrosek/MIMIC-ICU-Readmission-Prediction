"""
    Saving relevant things.
"""
import csv
import json

import numpy as np
import torch

def save_metrics(metrics_hist_all, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:

        data = metrics_hist_all[0].copy()

        data.update({"%s_te" % (name):val for (name,val) in metrics_hist_all[1].items()})
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_all[2].items()})

        json.dump(data, metrics_file, indent=1)

def save_params_dict(params):
    if "model_dir" in params.keys():
        with open(params["model_dir"] + "/params.json", 'w') as params_file:
            json.dump(params, params_file, indent=1)
    else:
        print("****no model dir given. not saving params****")

#def write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw=None):
def write_preds(yhat, model_dir, hids, fold, yhat_raw=None):
    """
        INPUTS:
            yhat: binary predictions matrix
            model_dir: which directory to save in
            hids: list of hadm_id's to save along with predictions
            fold: train, dev, or test
            yhat_raw: predicted scores matrix (floats)
    """

    preds_file = "%s/preds_%s.csv" % (model_dir, fold)
    with open(preds_file, 'w') as f:
        w = csv.writer(f, delimiter=',')
        for yhat_, hid in zip(yhat, hids):
            w.writerow([hid] + [yhat_])

def save_everything(args, metrics_hist_all, model, model_dir, params, criterion):
    """
        Save metrics, model, params all in model_dir
    """
    if args.test_model:
        return

    save_metrics(metrics_hist_all, model_dir)

    params['model_dir'] = model_dir
    save_params_dict(params)

    #save the model for best metrics
    if not np.all(np.isnan(metrics_hist_all[0][criterion])):
        if np.nanargmax(metrics_hist_all[0][criterion]) == len(metrics_hist_all[0][criterion]) - 1:
            torch.save(model, model_dir + "/model_best_%s.pth" % criterion)
    print("saved metrics, params, model to directory %s" % (model_dir))


