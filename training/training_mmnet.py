"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
import torch.optim as optim
from torch.autograd import Variable
import csv
import argparse
import os 
import numpy as np
import sys
import time
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd

# Adding relative path to python path
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)
sys.path.append(parent_dir)

from constants import *
from datasets import datasets
from evaluation import evaluation
from persistence import persistence
from tools import tools

def main(args):
    start = time.time()
    args, model, optimizer, params, dicts, struc_feats, struc_labels = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts, struc_feats, struc_labels)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize) # Sets field size to max available for strings

    # LOAD VOCAB DICTS
    dicts = datasets.load_vocab_dict(args.vocab_path)
    
    ## Loading structured data --> need to figure out best way to do this
    X, y = load_svmlight_file(args.struc_data_path)
    print("struc data loaded")
    
    num_struc_feats = X.shape[1]

    model = tools.pick_model(args, dicts, num_struc_feats)
    print(model)
    
    print("\nGPU: " + str(args.gpu))

    optimizer = optim.Adam(model.params_to_optimize(), weight_decay=args.weight_decay, lr=args.lr)

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts, X, y

def train_epochs(args, model, optimizer, params, dicts, struc_feats, struc_labels):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    
    print("\n\ntest_only: " + str(test_only))
        
    # Converting to csr sparse matrix form
    X = struc_feats.tocsr()
        
    # Splitting into train, val and test --> need idx values passed as args
    X_train = X[ : args.len_train]
    y_train = struc_labels[ : args.len_train]
    
    X_val = X[args.len_train : args.len_train + args.len_val]
    X_test = X[args.len_train + args.len_val : args.len_train + args.len_val + args.len_test]
            
    # Standardizing features
    scaler = MaxAbsScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    ################################

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
        metrics_all = one_epoch(model, optimizer, epoch, args.n_epochs, args.batch_size, args.data_path, test_only, dicts, model_dir, 
                                args.gpu, args.quiet, X_train_std, X_val_std, X_test_std, y_train, args.train_frac, args.test_frac, 
                                opt_thresh, args.struc_aux_loss_wt, args.conv_aux_loss_wt, args)
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
        
        if test_only or (epoch == args.n_epochs - 1):
            params['test_f1'] = metrics_hist_te['f1_micro'][0]
            params['test_auc'] = metrics_hist_te['auc'][0]

        metric_df = pd.DataFrame(params)
        metric_df.to_csv(model_dir + "/results.csv", index=False)

        #save metrics, model, params 
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion) 

        if test_only:
            break
        
        if (epoch == args.n_epochs - 2):
            opt_thresh = metrics_hist["opt_f1_thresh_micro"][np.nanargmax(metrics_hist[args.criterion])]
            print("Optimal f1 threshold: " + str(opt_thresh))

        if args.criterion in metrics_hist.keys():
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
        
def one_epoch(model, optimizer, epoch, n_epochs, batch_size, data_path, testing_only, dicts, model_dir, gpu, quiet, 
              struc_feats_train_std, struc_feats_val_std, struc_feats_test_std, struc_labels_train, train_frac, 
              test_frac, opt_thresh, struc_aux_loss_wt, conv_aux_loss_wt, args):
    """
        Basically a wrapper to do a training epoch and test on dev
    """
    if not testing_only:        
        
        losses = train(model, optimizer, epoch, batch_size, data_path, struc_feats_train_std, struc_labels_train, 
                       gpu, dicts, quiet, struc_aux_loss_wt, conv_aux_loss_wt, train_frac)
        loss = np.float64(np.mean(losses))
        print("epoch loss: " + str(loss))
        
    else:
        loss = np.nan

    pred_fold = "val" # fold to predict on

    metrics = test(model, epoch, batch_size, data_path, struc_feats_val_std, pred_fold, gpu, dicts, model_dir, testing_only, test_frac)
    
    # Predicting on test set using threshold learned on validation set
    opt_thresh = metrics["opt_f1_thresh_micro"]
    print("Optimal f1 threshold: " + str(opt_thresh))
            
    metrics_te = test(model, epoch, batch_size, data_path, struc_feats_test_std, "test", gpu, dicts, model_dir, True, test_frac, opt_thresh)
        
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    
    return metrics_all

def train(model, optimizer, epoch, batch_size, data_path, struc_feats, struc_labels, gpu, dicts, quiet, struc_aux_loss_wt, conv_aux_loss_wt, train_frac=1): ### struc feats = full structured sparse matrix
    """
        Training loop.
        output: losses for each example for this iteration
    """
    losses = []
    
    data_path = data_path[:-4] + "_reversed.csv"
    
    #how often to print some info to stdout
    print_interval = 50

    model.train() # PUTS MODEL IN TRAIN MODE
                   
    gen = datasets.data_generator(data_path, dicts, batch_size)
    for batch_idx, tup in enumerate(gen):
        
        if batch_idx * batch_size > train_frac * struc_feats.shape[0]:
            print("Reached {} of train set".format(train_frac))
            break
                
        data, target, hadm = tup
        
        batch_size_safe = min(batch_size, struc_feats.shape[0] - batch_idx * batch_size) # Avoiding going out of range

        struc_data = struc_feats[batch_idx * batch_size : batch_idx * batch_size + batch_size_safe].todense()
        struc_labels_batch = struc_labels[batch_idx * batch_size: batch_idx * batch_size + batch_size_safe] ### CAN USE THIS TO CONFIRM THAT LABELS MATCH BW STRUC AND TEXT
        
        if np.sum(target == struc_labels_batch) != batch_size_safe:
            print("Labels wrong, mismatch indices")
            print(batch_idx)
            print("-----------------")
            print("target")
            print(target)
            print("------------------")
            print("struc labels")
            print(struc_labels)
            break    
                    
        data, struc_data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(struc_data)), Variable(torch.FloatTensor(target).resize_(target.shape[0], 1))
        
        if gpu:
            data = data.cuda()
            struc_data = struc_data.cuda()
            target = target.cuda()
          
        optimizer.zero_grad()
                
        output, main_loss, struc_aux_loss, conv_aux_loss = model(data, struc_data, target) # FORWARD PASS
        
        loss = main_loss + struc_aux_loss * struc_aux_loss_wt + conv_aux_loss * conv_aux_loss_wt

        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
        
        if not quiet and batch_idx % print_interval == 0:
            #print the average loss of the last 100 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-100:])))
            print("Main loss, struc loss, conv loss: " + str((main_loss.data[0], struc_aux_loss.data[0], conv_aux_loss.data[0])))

    return losses


def test(model, epoch, batch_size, data_path, struc_feats, fold, gpu, dicts, model_dir, testing, test_frac = 1, thresh=None):

    """
        Testing loop.
        Returns metrics
    """
    filename = data_path.replace('train', fold)
    filename = filename[:-4] + "_reversed.csv"
    print('\nfile for evaluation: %s' % filename)
    
    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
        
    model.eval()
    gen = datasets.data_generator(filename, dicts, batch_size)
    for batch_idx, tup in enumerate(gen):
        
        data, target, hadm_ids = tup
        
        batch_size_safe = min(batch_size, struc_feats.shape[0] - batch_idx * batch_size) # Avoiding going out of range
                             
        if batch_idx * batch_size > test_frac * struc_feats.shape[0]:
            print("Reached {} of test/val set".format(test_frac))
            break
                             
        struc_data = struc_feats[batch_idx * batch_size: batch_idx * batch_size + batch_size_safe].todense() # Only need in second index b/c batch_size_safe should be < batch_size only once
                    
        data, struc_data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(struc_data)), Variable(torch.FloatTensor(target))
        
        if gpu:
            data = data.cuda()
            struc_data = struc_data.cuda()
            target = target.cuda()
            
        model.zero_grad()

        output, main_loss, _ , _ = model(data, struc_data, target) # Forward pass

        output = output.data.cpu().numpy()
        output_safe = np.clip(output, -100, 100) # To ensure numerical stability
        output = 1/(1+np.exp(-1 * output_safe)) 
                
        losses.append(main_loss.data[0]) 
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
    parser.add_argument("struc_data_path", type=str,
                        help="path to a file containing sorted structured train data")
    parser.add_argument("vocab_path", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("model", type=str, choices=["conv_encoder", "mmnet", "saved"], help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    
    parser.add_argument("len_train", type=int, help="number of observations in training set")
    parser.add_argument("len_val", type=int, help="number of observations in validation set")
    parser.add_argument("len_test", type=int, help="number of observations in the test set")
    
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--loss", type=str, required=False, dest="loss", default = "BCE",
                        help="Loss function to use, either BCE or margin_ranking_loss")
    parser.add_argument("--bce-weights", type=str, required=False, dest="bce_weights", default = None,
                        help="Weights applied to negative and positive classes respectively for Binary Cross entropy loss. Ex: 0.1, 1 --> 10x more weight to positive instances")
    
    # Text
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
                        help="optional specification of dropout proportion for fully connected layers that receive feature maps as input")
    parser.add_argument("--embed-dropout-p", dest="embed_dropout_p", type=float, required=False, default=0.2,
                        help="optional specification of dropout proportion for embedding layer")
    parser.add_argument("--embed-dropout-bool", dest="embed_dropout_bool", type=bool, required=False, default = False,
                        help="optional specification of whether to employ dropout on embedding layer")
    
    parser.add_argument("--post-conv-fc-bool", dest="post_conv_fc_bool", type=bool, required=False, default = False,
                        help="optional specification of whether to add a hidden layer after convolving over text")
    parser.add_argument("--post-conv-fc-dim", dest="post_conv_fc_dim", type=int, required=False, default = 16,
                        help="size of post-conv fc layer")
    parser.add_argument("--conv-aux-loss-wt", type=float, required=False, dest="conv_aux_loss_wt", default=0,
                        help="Weight to give to the auxiliary loss placed at end of convolutional/text branch (main loss has weight 1.0)")
        
    
    # Structured
    parser.add_argument("--struc-layer-size-list", type=str, required=False, dest="struc_layer_size_list", default=3,
                        help="Number of units in each hidden layer Ex: 3,4,5)")
    parser.add_argument("--struc-activation", type=str, required=False, dest="struc_activation", default="selu",
                        help="non-linear activation to be applied to fc layers. Must match PyTorch documentation for torch.nn.functional.[conv_activation]")
    parser.add_argument("--struc-dropout-list", type=str, required=False, dest="struc_dropout_list", default=None,
                        help="Dropout proportion on each hidden layer in structured branch. First number assumed to correspond to first hidden layer. Ex: 0.5,0.1")
    parser.add_argument("--struc-aux-loss-wt", type=float, required=False, dest="struc_aux_loss_wt", default=0,
                        help="Weight to give to the auxiliary loss placed at end of structure branch (main loss has weight 1.0)")
    
    
    # Post Merge
    parser.add_argument("--post-merge-layer-size-list", type=str, required=False, dest="post_merge_layer_size_list", default=None,
                        help="Number of units in each hidden layer Ex: 3,4,5)")
    parser.add_argument("--batch-norm-bool", dest="batch_norm_bool", type=bool, required=False, default = False,
                        help="whether to add a batch normalization layer after concatenating vectors from structured and text branches")
    
    # For debugging can run on fraction of data
    parser.add_argument("--train-frac", type=float, help="fraction of training split to train on", required=False, dest="train_frac", default=1.0)
    parser.add_argument("--test-frac", type=float, help="fraction of test split to test on", required=False, dest="test_frac", default=1.0)
   
    
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
    
    args = parser.parse_args()
    
    if args.struc_dropout_list == '':
        args.struc_dropout_list = None
    
    print("post merge hidden layer sizes: " + str(args.post_merge_layer_size_list))
    print("post convolution fc layer(s)?: " + str(args.post_conv_fc_bool))
    print("batch norm?: " + str(args.batch_norm_bool))
    print("struc_aux_loss_wt: " + str(args.struc_aux_loss_wt))
    print("conv_aux_loss_wt: " + str(args.conv_aux_loss_wt))
    
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)

