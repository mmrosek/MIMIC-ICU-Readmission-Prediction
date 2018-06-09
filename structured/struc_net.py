import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import json
from collections import defaultdict
import datetime
import os
from torch.nn.init import xavier_uniform
import pandas as pd

# Adding relative path to python path
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)
sys.path.append(parent_dir)

from evaluation import evaluation as evaluation

# Need to pass array of hidden layer sizes

class WeightedBCELoss(nn.modules.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
    def forward(self, y_pred, target, weights=None):
        if weights is not None:
            assert len(weights) == 2

            #print(y_pred)
            #print(target)

            #print("first term: "+ str(float(weights[1]) * (target * torch.log(y_pred))))
            #print("second term: "+ str(float(weights[0]) * ((1 - target) * torch.log(1 - y_pred))))

            weight = float(weights[1])/float(weights[0])

            loss = F.binary_cross_entropy_with_logits(y_pred, target, weight)

            #print(loss)

            #return torch.neg(torch.mean(loss))
            return loss

        else:

            return F.binary_cross_entropy_with_logits(y_pred, target)


class FeedForwardNet(nn.Module):
    def __init__(self, n_input, n_output, fc_layer_size_list, fc_dropout_list, fc_activation):

        super(FeedForwardNet, self).__init__()
        self.fc_layer_size_list = fc_layer_size_list
        self.fc_dropout_list = fc_dropout_list
        self.output = nn.Linear(self.fc_layer_size_list[len(self.fc_layer_size_list)-1], n_output)

        # Initializing dropouts on each hidden layer
        self.fc_dropouts = [nn.Dropout(p = fc_dropout_list[idx]) for idx in range(len(fc_dropout_list))]

        if len(self.fc_dropouts) > 0: # If employing dropout..
            for idx, fc_dropout in enumerate(self.fc_dropouts):
                self.add_module('dropout_%d' % idx, fc_dropout)

        # Initializing fully-connected layers layers
        self.fc_layer_size_list.insert(0, n_input) # Inserting num_inputs to serve as input dim for first hidden layer

        self.fc_layers = [nn.Linear(self.fc_layer_size_list[idx], self.fc_layer_size_list[idx+1]) for idx in range(len(self.fc_layer_size_list) - 1)]
        for idx, fc_layer in enumerate(self.fc_layers):
            self.add_module('fc_%d' % idx, fc_layer)

            # NEED TO INVESTIGATE
            ff_layer = getattr(self, 'fc_{}'.format(idx)) # ~ self.fc_i
            xavier_uniform(ff_layer.weight)

        # Setting non-linear activation on fc layers
        self.fc_activation = getattr(F, fc_activation) # Equivalent to F.[fc_activation]

    def forward(self, x):

        for i in range(len(self.fc_layers)): # Passing input through each fully connected layer

            fc_layer = getattr(self, 'fc_{}'.format(i)) # ~ self.fc_i

            if i <= (len(self.fc_dropout_list) - 1) : # If dropout applied to this layer (assuming first value in fc_dropout_list is for first layer)

                dropout = getattr(self, 'dropout_{}'.format(i)) # ~ self.dropout_i
                x = dropout(self.fc_activation(fc_layer(x)))
            else:
                x = self.fc_activation(fc_layer(x))

        x = self.output(x) # Prediction

        return x


def train_and_test_net(X, y, fc_network, criterion, optimizer, num_epochs, batchsize, train_frac, test_frac, gpu_bool, bce_weights):

    # Converting to csr sparse matrix form
    X = X.tocsr()

    # Splitting into train, val and test
    X_train = X[:-5273]
    X_val = X[-5273:-2637]
    X_test = X[-2637:]

    y_train = y[:-5273]
    y_val = y[-5273:-2637]
    y_test = y[-2637:]

    # Standardizing features
    scaler = MaxAbsScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    df_metrics = pd.DataFrame(columns=['epoch', 'timestamp', 'batchsize', 'fc_activation', 'hidden_layers', 'test_frac', 'train_frac', 'dropout', 'train_loss', 'val_loss', 'opt_thresh', 'f1', 'auc', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'test_opt_thresh', 'test_f1', 'test_auc', 'test_true_pos', 'test_true_neg', 'test_false_pos', 'test_false_neg', 'bce_weights'])
    ### Training Set ###

    print("starting training")
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_no = epoch
        train_loss = 0
        val_loss = 0
        opt_thresh = 0
        train_loss = 0
        val_loss = 0
        opt_thresh = 0
        f1 = 0
        auc = 0
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        test_opt_thresh = 0
        test_f1 = 0
        test_auc = 0
        test_true_pos = 0
        test_true_neg = 0
        test_false_pos = 0
        test_false_neg = 0


        running_loss = 0.0 # Loss over a set of mini batches

        i = 0
        k = 0
        losses = []

        print("Number of training examples: " + str(int(X_train.shape[0]*train_frac)))

        while i < X_train.shape[0]*train_frac:

            batch_size_safe = min(batchsize, X_train.shape[0] - i) # Avoiding going out of range

            Xtrainsample = X_train_std[i:i+batch_size_safe].todense()
            ytrainsample = y_train[i:i+batch_size_safe]

            inputs = torch.from_numpy(Xtrainsample.astype('float32'))
            labels = torch.from_numpy(ytrainsample.astype('float32')).view(-1,1)

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            if gpu_bool:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = fc_network(inputs)
            loss = criterion(outputs, labels, bce_weights)

            # backward
            loss.backward()
            losses.append(loss.data)

            # optimize
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            if k % 100 == 99:    # print every 100 mini-batches
                print('[epoch:%d, batch:%5d] loss over last 100 batches: %.3f' %
                      (epoch + 1, k + 1, running_loss / 100))
                running_loss = 0.0

            k = k+1
            i = i+batchsize

        #metrics_dict["tr_loss_{}".format(epoch)] = np.mean(losses)
        train_loss = np.mean(losses)

        print('Finished Epoch')
        print("Predicting on validation set")

        ### Validation Set ###

        y_true = []
        y_pred = []
        losses = []

        i = 0

        while i < X_val_std.shape[0]*test_frac:

            batch_size_safe = min(batchsize, X_val_std.shape[0] - i) # Avoiding going out of range

            Xtestsample = X_val_std[i:i+batch_size_safe].todense()
            ytestsample = y_val[i:i+batch_size_safe]

            inputs = torch.from_numpy(Xtestsample.astype('float32'))
            labels = torch.from_numpy(ytestsample.astype('float32')).view(-1,1)

            if gpu_bool:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = fc_network(Variable(inputs))

            loss = criterion(outputs, Variable(labels), bce_weights)
            losses.append(loss.data)

            outputs = F.sigmoid(outputs)


            # Converting to numpy format
            outputs = outputs.data.cpu().numpy()
            labels = labels.cpu().numpy()

            y_true.extend(labels.flatten().tolist())
            y_pred.extend(outputs.flatten().tolist())
            i = i + batchsize

        print("finished predicting on validation set")

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        val_loss = np.mean(losses)

        f1, opt_thresh = evaluation.find_opt_thresh_f1(y_pred, y_true, 0.01, 0.5, 50)
        auc_metrics = evaluation.auc_metrics(y_pred, y_true, opt_thresh)
        auc = auc_metrics["auc"]
        true_pos = auc_metrics["true_pos"]
        true_neg = auc_metrics["true_neg"]
        false_pos = auc_metrics["false_pos"]
        false_neg = auc_metrics["false_neg"]
        print("AUC on epoch {}: ".format(epoch+1) + str(auc))
        print("F1 on epoch {}: ".format(epoch+1) + str(f1))
        print("opt f1 thresh on epoch {}: ".format(epoch+1) + str(opt_thresh))

        ### Test Set ###

        y_true = []
        y_pred = []
        i = 0

        while i < X_test_std.shape[0]*test_frac:

            batch_size_safe = min(batchsize, X_test_std.shape[0] - i) # Avoiding going out of range

            Xtestsample = X_test_std[i:i+batch_size_safe].todense()
            ytestsample = y_test[i:i+batch_size_safe]

            inputs = torch.from_numpy(Xtestsample.astype('float32'))
            #labels = torch.from_numpy(ytestsample.astype('float32')).view(-1,1)

            if gpu_bool:
                inputs = inputs.cuda()

            outputs = fc_network(Variable(inputs))
            outputs = F.sigmoid(outputs)

            # Converting to numpy format
            outputs = outputs.data.cpu().numpy()

            y_true.extend(np.array(ytestsample).tolist())
            y_pred.extend(outputs.flatten().tolist())
            i = i + batchsize

        print("\nfinished predicting on test set")

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        #epoch_key = "epoch_{}".format(epoch+1)

        #metrics_dict[epoch_key]["test_f1"], metrics_dict[epoch_key]["test_opt_thresh"] = evaluation.find_opt_thresh_f1(y_pred, y_true, 0.01, 0.5, 50)
        #metrics_dict[epoch_key]["test_auc"] = evaluation.auc_metrics(y_pred, y_true, metrics_dict[epoch_key]["test_opt_thresh"]) # auc_metrics() returns a dictionary
        #print("test AUC on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["test_auc"]))
        #print("test F1 on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["test_f1"]))
        #print("test opt f1 thresh on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["test_opt_thresh"]))

        test_f1, test_opt_thresh = evaluation.find_opt_thresh_f1(y_pred, y_true, 0.01, 0.5, 50)
        test_auc_metrics = evaluation.auc_metrics(y_pred, y_true, opt_thresh)
        test_auc = test_auc_metrics["auc"]
        test_true_pos = test_auc_metrics["true_pos"]
        test_true_neg = test_auc_metrics["true_neg"]
        test_false_pos = test_auc_metrics["false_pos"]
        test_false_neg = test_auc_metrics["false_neg"]
        print("Test AUC on epoch {}: ".format(epoch+1) + str(test_auc))
        print("Test F1 on epoch {}: ".format(epoch+1) + str(test_f1))
        print("Test opt f1 thresh on epoch {}: ".format(epoch+1) + str(test_opt_thresh))

        df_metrics = df_metrics.append({'epoch': epoch_no+1, 'train_loss': train_loss, 'val_loss': val_loss, 'opt_thresh': opt_thresh, 'f1': f1, 'auc': auc, 'true_pos': true_pos, 'true_neg': true_neg, 'false_pos': false_pos, 'false_neg': false_neg, 'test_opt_thresh': test_opt_thresh, 'test_f1': test_f1, 'test_auc': test_auc, 'test_true_pos': test_true_pos, 'test_true_neg': test_true_neg, 'test_false_pos': test_false_pos, 'test_false_neg': test_false_neg}, ignore_index=True)


    return df_metrics


def main(data_path, fc_layer_size_list, fc_dropout_list, fc_activation, num_epochs, batch_size, train_frac, test_frac, gpu_bool, bce_weights):

    # Loading data
    X, y = load_svmlight_file(data_path)
    print("data loaded")

    net = FeedForwardNet(n_input=X.shape[1], n_output=1, fc_layer_size_list=fc_layer_size_list, fc_dropout_list=fc_dropout_list, fc_activation = fc_activation)
    print("defined network")

    print("\nGPU: " + str(gpu_bool))

    if gpu_bool:
        net.cuda()

    criterion = WeightedBCELoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0, lr=0.001)

    df_metrics = train_and_test_net(X, y, net, criterion, optimizer, num_epochs, batch_size, train_frac, test_frac, gpu_bool, bce_weights)

    return  df_metrics


if __name__ == "__main__":

    current_time = datetime.datetime.now()
    month = str(current_time).split('-')[1]
    day = str(current_time).split('-')[2][:-10].split()[0]
    mins = str(current_time).split('-')[2][:-10].split()[1]
    date =  month + '_' + day + "_" + mins

    running_ide = False

    if running_ide == True:

        data_path = "/home/miller/Documents/BDH NLP/Data/"
#
        auc_dict, f1_dict = main(data_path, [1024], [0.6] , "relu", 1, 32)

    else:

        parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
        parser.add_argument("data_path", type=str,
                            help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
        parser.add_argument("write_path", type=str,
                            help="path to write file to")
        parser.add_argument("num_epochs", type=int, help="number of epochs to train")
        parser.add_argument("--train-frac", type=float, help="fraction of training split to train on", required=False, dest="train_frac", default=1.0)
        parser.add_argument("--test-frac", type=float, help="fraction of test split to test on", required=False, dest="test_frac", default=1.0)
        parser.add_argument("--bce-weights", type=str, required=False, dest="bce_weights", default = None,
                            help="Weights applied to negative and positive classes respectively for Binary Cross entropy loss. Ex: 0.1, 1 --> 10x more weight to positive instances")
        parser.add_argument("--fc-layer-size-list", type=str, required=False, dest="fc_layer_size_list", default=3,
                            help="Number of units in each hidden layer Ex: 3,4,5)")
        parser.add_argument("--fc-activation", type=str, required=False, dest="fc_activation", default="selu",
                            help="non-linear activation to be applied to fc layers. Must match PyTorch documentation for torch.nn.functional.[conv_activation]")
        parser.add_argument("--dropout-list", type=str, required=False, dest="dropout_list", default=None,
                            help=" Dropout proportion on each hidden layer. First number assumed to correspond to first hidden layer. Ex: 0.5,0.1")
        parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=32,
                            help="size of training batches")
        parser.add_argument("--patience", type=int, default=2, required=False,
                            help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
        parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                            help="optional flag to use GPU if available")
        args = parser.parse_args()
        command = ' '.join(['python'] + sys.argv)
        args.command = command

        if args.dropout_list:
            dropouts = [float(size) for size in args.dropout_list.split(",")]
        else:
            dropouts = []

        fc_layer_sizes = [int(size) for size in args.fc_layer_size_list.split(",")]

        bce_weights = [float(weights) for weights in args.bce_weights.split(",")]

        print(fc_layer_sizes)

        df_metrics = main(args.data_path, fc_layer_sizes, dropouts, args.fc_activation, args.num_epochs, args.batch_size, args.train_frac, args.test_frac, args.gpu, bce_weights)

        # params = defaultdict(list)
        # params["dropout"] = dropouts
        # params["hidden_layers"] = fc_layer_sizes[1:] # Number of inputs gets inserted at 0th index in main function
        # params["train_frac"] = args.train_frac
        # params["test_frac"] = args.test_frac

        # metrics_dict.update(params)

        df_metrics['timestamp'] = date
        df_metrics['hidden_layers'] = ','.join(map(str, fc_layer_sizes[1:]))
        df_metrics['dropout'] = ','.join(map(str, dropouts))
        df_metrics['train_frac'] = args.train_frac
        df_metrics['test_frac'] = args.test_frac
        df_metrics['fc_activation'] = args.fc_activation
        df_metrics['batchsize'] = args.batch_size
        df_metrics['bce_weights'] = ','.join(map(str, bce_weights))

        print(df_metrics)

        #with open(args.write_path + "fc_results_{}.txt".format(date), "w") as text_file:
            #text_file.write(str(metrics_dict))
        df_metrics.to_csv(args.write_path + "new_bce_fc_results_{}.csv".format(date), index=False)



