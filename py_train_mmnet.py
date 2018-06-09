import subprocess
import shlex
import random
import numpy as np
import pandas as pd
import math

data_path = "../../../Data/"

train = pd.read_csv(data_path + "sorted_sums_matched_struc_train_reversed.csv")
val = pd.read_csv(data_path + "sorted_sums_matched_struc_val_reversed.csv")
test = pd.read_csv(data_path + "sorted_sums_matched_struc_test_reversed.csv")

len_train = len(train)
len_val = len(val)
len_test = len(test)

del train, val, test

kernel_sizes_str = ""
number_filter_maps = ""
fc_dropout = ""
embed_dropout = ""
batch_size = ""
weight_decay = ""

#Randomizing number of kernels and kernel sizes

for i in range(0, 10000):

    print("=======================Iteration: "+str(i) + " ===============================")

    number_kernels = random.randint(2, 3)

    kernel_sizes_distribution = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9]

    kernel_sizes = random.sample(range(2, 9), number_kernels)

    kernel_sizes_str = ','.join(map(str, kernel_sizes))


    #Randomizing number of features
    number_filter_maps = str(random.randint(100, 500))


    #Fully connected dropout
    fc_dropout = str(random.uniform(0.1, 0.6))


    #Randomizing embed dropout
    embed_dropout = str(random.uniform(0, 0.5))

    #Randomizing auxiliary losses
    struc_aux_loss_wt_str = str(random.uniform(0, 0.3) * random.randint(0,2))
    conv_aux_loss_wt_str = str(random.uniform(0, 0.3) * random.randint(0,2))


    #Randomizing batch sizes
    batch_size_list = [8,16]
    batch_size = str(batch_size_list[random.randint(0, 1)])


    #Randomizing bce weights
    bce_weights_list = [1, 0.8, 0.5, 0.4, 0.31, 0.29, 0.27, 0.23, 0.2, 0.18, 0.16, 0.15]

    bce_weight_str = str(bce_weights_list[random.randint(0, len(bce_weights_list)-1)]) + ',1'

    weight_decay_list = [0,0,0,0,0,0,0]

    weight_decay_str = str(weight_decay_list[random.randint(0, len(weight_decay_list)-1)])


    print("kernel_sizes_str: " + kernel_sizes_str)
    print("number_filter_maps: " + number_filter_maps)
    print("fc_dropout: " + fc_dropout)
    print("embed_dropout: " + embed_dropout)
    print("batch_size: " + batch_size)
    print("bce_weight_str: " + bce_weight_str)
    print("weight_decay: " + weight_decay_str)
    
    
    #### STRUCTURED ####
    fc_layer_sizes_str = ""
    embed_dropout_list_str = ""
    activation_layer = ""
    fc_layer_size_list = random.randint(1, 3)
    fc_layer_sizes = []
    for i in range(0, fc_layer_size_list):
        if (i == 0):
            fc_layer_sizes.append(random.randint(16, 512))
        elif (i == 1):
            fc_layer_sizes.append(random.randint(8, math.ceil(0.5*fc_layer_sizes[0])))
        else:
            fc_layer_sizes.append(random.randint(4, math.ceil(0.5*fc_layer_sizes[1])))

    fc_layer_sizes_str = ','.join(map(str, fc_layer_sizes))

    activation_layer_list = ["relu", "selu"]
    activation_layer = activation_layer_list[random.randint(0, len(activation_layer_list) - 1)]

    struc_dropout_list_str = str(random.uniform(0, 0.6))
 

    print("fc_layer_sizes_str: " + fc_layer_sizes_str)
    print("activation_layer: " + activation_layer)
    print("embed_dropout_list_str: " + embed_dropout_list_str)


    #Batch Normalization
    batch_norm_bool = random.uniform(0,1)

    if batch_norm_bool > 0.7: # If batch norm being used:

	    var_list = ["./train.sh",                   # Shell script

		        "training/training_mmnet.py",        # Training script

		        "../../../Data/sorted_sums_matched_struc_train.csv", # Data path
                        
                "../../../Data/struc_data_reversed.svmlight",

		        "../../../Data/vocab.csv",         # Vocab path

		        "mmnet",                    # Model

		        "5",                               # Num epochs
                
                	str(len_train),
                   
                	str(len_val),
                   
                	str(len_test),

		        "--embed-dropout-bool",

		        "True",

		        "--embed-dropout-p",

		        embed_dropout,

		        "--bce-weights",

		        bce_weight_str,

		        "--batch-size",

		        batch_size,

		        "--kernel-sizes",

		        kernel_sizes_str,

		        "--num-filter-maps",

		        number_filter_maps,

		        "--fc-dropout-p",

		        fc_dropout,

		        "--embed-file",

		        "../../../Data/processed_full.embed",

		        "--gpu",

		        "--weight-decay",

		        weight_decay_str,

		        "--batch-norm-bool",

		        "True",

                	"--conv-aux-loss-wt",

                	conv_aux_loss_wt_str,

    			"--struc-aux-loss-wt",

                	struc_aux_loss_wt_str,
                
                	"--struc-layer-size-list",

                	fc_layer_sizes_str,

                	"--struc-activation",

                	activation_layer,

                	"--struc-dropout-list",

                	struc_dropout_list_str]

    else:

	    var_list = ["./train.sh",                   # Shell script

                "training/training_mmnet.py",        # Training script

                "../../../Data/sorted_sums_matched_struc_train.csv", # Data path

                "../../../Data/struc_data_reversed.svmlight",

                "../../../Data/vocab.csv",         # Vocab path

                "mmnet",                           # Model

                "5",                               # Num epochs
                
                str(len_train),
                str(len_val),
                str(len_test),

                "--embed-dropout-bool",

                "True",

                "--embed-dropout-p",

                embed_dropout,

                "--bce-weights",

                bce_weight_str,

                "--batch-size",

                batch_size,

                "--kernel-sizes",

                kernel_sizes_str,

                "--num-filter-maps",

                number_filter_maps,

                "--fc-dropout-p",

                fc_dropout,

                "--embed-file",

                "../../../Data/processed_full.embed",

                "--gpu",

                "--weight-decay",

                weight_decay_str,

        	"--conv-aux-loss-wt",

                conv_aux_loss_wt_str,

        	"--struc-aux-loss-wt",

                struc_aux_loss_wt_str,
                
                "--struc-layer-size-list",
                
		fc_layer_sizes_str,
                
		"--struc-activation",
                
		activation_layer,
                
		"--struc-dropout-list",
                
		struc_dropout_list_str]



    string = ""


    for idx in range(len(var_list)):


        if idx < (len(var_list) - 1):


            string = string + var_list[idx] + ' '


        else:


            string += var_list[idx]





    subprocess.call(shlex.split(string))

