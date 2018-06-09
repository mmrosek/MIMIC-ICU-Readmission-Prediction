"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import csv
import argparse
import os 
import numpy as np
import sys
import time
from collections import defaultdict
from functools import reduce
import gc
import operator as op

# Adding relative path to python path
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)
sys.path.append(parent_dir)

from constants import *
from datasets import datasets
from evaluation import evaluation
from persistence import persistence
from tools import tools_vM1 as tools

def main(args):
    start = time.time()
    args, model, optimizer, params, dicts = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize) # Sets field size to max available for strings

    # LOAD VOCAB DICTS
    dicts = datasets.load_vocab_dict(args.vocab_path)

    model = tools.pick_model(args, dicts)
    print(model)
    
    print("\nGPU: " + str(args.gpu))

    optimizer = optim.Adam(model.params_to_optimize(), weight_decay=args.weight_decay, lr=args.lr)
    
    print("Weight decay: " + str(args.weight_decay))

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts

def train_epochs(args, model, optimizer, params, dicts):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    
    print("\n\ntest_only: " + str(test_only))
    
    opt_thresh = None # Placeholder, only needed when predicting on test set, updated below
        
    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        
        #only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, args.desc, time.strftime('%b_%d_%H:%M', time.gmtime())]))
            os.mkdir(model_dir) 
            
        elif args.test_model:
            
            model_dir = os.getcwd() #just save things to where this script was called
        
        start = time.time()
        metrics_all = one_epoch(model, optimizer, epoch, args.n_epochs, args.batch_size, args.data_path,test_only, dicts, model_dir, 
                                                  args.gpu, args.quiet, opt_thresh, args.obs_limit)
        end = time.time()
        print("\nEpoch Duration: " + str(end-start))

        # DISTRIBUTING results from metrics_all to respective dicts
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
                       
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
            
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
            
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)
            
        
        ### Writing to csv ###
        params['kernel_sizes'] = str(params['kernel_sizes'])            
        params['val_auc'] = metrics_hist['auc']
        params['val_f1'] = metrics_hist['f1_micro']
        
        if test_only or (epoch == args.n_epochs -1):
            params['test_f1'] = metrics_hist_te['f1_micro'][0]
            params['test_auc'] = metrics_hist_te['auc'][0]

        metric_df = pd.DataFrame(params)
        metric_df.to_csv(model_dir + "/results.csv", index=False)
            
        #save metrics, model, params
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion) # SHOULD SAVE MODEL PARAMS AT EACH EPOCH, BELIEVE IS HAPPENING

        if test_only:
            break

        if (epoch == args.n_epochs - 2):
            opt_thresh = metrics_hist["opt_f1_thresh_micro"][np.nanargmax(metrics_hist[args.criterion])]
            print("Optimal f1 threshold: " + str(opt_thresh))

        if (args.criterion in metrics_hist.keys()):
            if (early_stop(metrics_hist, args.criterion, args.patience)):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping or just completed last epoch" % (args.criterion, args.patience))
                test_only = True
                opt_thresh = metrics_hist["opt_f1_thresh_micro"][np.nanargmax(metrics_hist[args.criterion])]
                print("Optimal f1 threshold: " + str(opt_thresh))
                model = torch.load('%s/model_best_%s.pth' % (model_dir, args.criterion)) # LOADING BEST MODEL FOR FINAL TEST
                
    return epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if criterion == 'loss-dev': 
            
            ### EPOCH NUM W/ MIN DEV LOSS < (CURR EPOCH NUM - PATIENCE) ?? RETURNS ANSWER AS BOOL --> IF TRUE, STOP TRAINING
            ### EX: 5 < 9 - 3 = TRUE = EARLY STOP
            return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
        else:
            
            ### EPOCH NUM W/ MAX CRITERION VAL < (CURR EPOCH NUM - PATIENCE) ?? RETURNS ANSWER AS BOOL
            return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False
        
def one_epoch(model, optimizer, epoch, n_epochs, batch_size, data_path, testing_only, dicts, model_dir, gpu, quiet, opt_thresh, obs_limit):
    """
        Wrapper to perform a training epoch and test on validation or test set.
    """
    if not testing_only:        
        losses = train(model, optimizer, epoch, batch_size, data_path, gpu, dicts, quiet, obs_limit)
        loss = np.float64(np.mean(losses))
        print("epoch loss: " + str(loss))
        
    else:
        loss = np.nan

    pred_fold = "val" # fold to predict on

    metrics = test(model, epoch, batch_size, data_path, pred_fold, gpu, dicts, model_dir, testing_only, opt_thresh, obs_limit)
    
    if testing_only or epoch == n_epochs - 1:
        print("evaluating on test")
        metrics_te = test(model, epoch, batch_size, data_path, "test", gpu, dicts, model_dir, True, opt_thresh, obs_limit)

    else:
        metrics_te = defaultdict(float)
        
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    
    return metrics_all

def train(model, optimizer, epoch, batch_size, data_path, gpu, dicts, quiet, obs_limit = None):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    losses = []
    
    #how often to print some info to stdout
    print_interval = 50
    
    data_path = data_path[:-4] + "_reversed.csv"
    
    model.train() # PUTS MODEL IN TRAIN MODE
                   
    gen = datasets.data_generator(data_path, dicts, batch_size)
    for batch_idx, tup in enumerate(gen):
        
        if obs_limit: # Early stopping for debugging`
            obs_seen = batch_idx * batch_size 
            if obs_seen > obs_limit:
                print("Reached {} of test/val set".format(obs_limit))
                break

        data, target, hadm = tup
                
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        
        if gpu:
            data = data.cuda()
            target = target.cuda()
            
        optimizer.zero_grad()
        
        output, loss = model(data, target) # FORWARD PASS

        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])

        if not quiet and batch_idx % print_interval == 0:
            #print the average loss of the last 100 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-100:])))
            
            ### Printing memory
            total=0
            for obj in gc.get_objects():
                
                if torch.is_tensor(obj):
                    obj_size = reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0
#                    print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
                    total += obj_size
            print(total)

        del output, loss, data, target

    return losses


def test(model, epoch, batch_size, data_path, fold, gpu, dicts, model_dir, testing, thresh = None, obs_limit = None):

    """
        Testing loop. Returns metrics on desired fold (validation or test).
    """
    filename = data_path.replace('train', fold)
    filename = filename[:-4] + "_reversed.csv"
    print('\nfile for evaluation: %s' % filename)
    
    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    
    model.eval()
    gen = datasets.data_generator(filename, dicts, batch_size)
    for batch_idx, tup in enumerate(gen):
        
        if obs_limit: # Early stopping for debugging
            obs_seen = batch_idx * batch_size 
            if obs_seen > obs_limit:
                print("Reached {} of test/val set".format(obs_limit))
                break
        
        data, target, hadm_ids = tup
        
        data, target = Variable(torch.LongTensor(data), volatile=True), Variable(torch.FloatTensor(target))
        
        if gpu:
            data = data.cuda()
            target = target.cuda()
            
        model.zero_grad()

        output, loss = model(data, target) # Forward pass

        output = output.data.cpu().numpy()
        losses.append(loss.data[0]) 
        target_data = target.data.cpu().numpy()
        
        #save predictions, target, hadm ids
        yhat_raw.append(output) 
        output = np.round(output) # Rounds to 0 for <= 0.5, up to one for > 0.5
        yhat.append(output)
        
        y.append(target_data)
        hids.extend(hadm_ids)
    
    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)
    
    print("\nMax Prediction This Epoch:")
    print(max(yhat_raw))

    #write the predictions
    persistence.write_preds(yhat, model_dir, hids, fold, yhat_raw)
        
    metrics = evaluation.all_metrics(yhat_raw, y, thresh)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.float64(np.mean(losses)) #float64 for json serialization
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab_path", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("model", type=str, choices=["conv_encoder", "rnn", "conv_attn", "multi_conv_attn", "saved"], help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--loss", type=str, required=False, dest="loss", default = "BCE",
                        help="Loss function to use, either BCE or margin_ranking_loss")
    parser.add_argument("--bce-weights", type=str, required=False, dest="bce_weights", default = None,
                        help="Weights applied to negative and positive classes respectively for Binary Cross entropy loss. Ex: 0.1, 1 --> 10x more weight to positive instances")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--kernel-sizes", type=list, required=False, dest="kernel_sizes", default=3,
                        help="Size(s) of convolutional filter(s)/kernel(s) to use. Ex: 3,4,5)")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--conv-activation", type=str, required=False, dest="conv_activation", default="selu",
                        help="non-linear activation to be applied to feature maps. Must match PyTorch documentation for torch.nn.functional.[conv_activation]")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--fc-dropout-p", dest="fc_dropout_p", type=float, required=False, default=0.5,
                        help="optional specification of dropout proportion for fully connected layers")
    parser.add_argument("--embed-dropout-p", dest="embed_dropout_p", type=float, required=False, default=0.2,
                        help="optional specification of dropout proportion for embedding layer")
    parser.add_argument("--embed-dropout-bool", dest="embed_dropout_bool", type=bool, required=False,
                        help="optional specification of whether to employ dropout on embedding layer")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=2, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    parser.add_argument("--desc", dest="desc", type=str, required=False, default = '',
                        help="optional flag for description of training run")
    parser.add_argument("--obs-limit", type=float, help="Max number of observations to train/test on --> for debugging", required=False, dest="obs_limit", default=None)
    
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)


