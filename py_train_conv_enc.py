import subprocess
import shlex
import random
import numpy as np

kernel_sizes_str = ""
number_filter_maps = ""
fc_dropout = ""
embed_dropout = ""
batch_size = ""
weight_decay = ""

#Randomizing number of kernels and kernel sizes

for i in range(0, 10000):

    print("=======================Iteration: "+str(i) + " ===============================")

    number_kernels = random.randint(2, 4)

    kernel_sizes_distribution = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9]

    kernel_sizes = random.sample(range(2, 9), number_kernels)

    kernel_sizes_str = ','.join(map(str, kernel_sizes))

    #Randomizing number of feature maps
    number_filter_maps = str(random.randint(100, 500))


    #Fully connected dropout
    fc_dropout = str(random.uniform(0.1, 0.6))


    #Randomizing embed dropout
    embed_dropout = str(random.uniform(0, 0.5))


    #Randomizing batch sizes
    batch_size_list = [8,16]

    batch_size = str(batch_size_list[random.randint(0, 1)])


    #Randomizing bce weights
    bce_weights_list = [1, 1, 0.8, 0.5, 0.4, 0.31, 0.29, 0.27, 0.23, 0.2, 0.18, 0.16, 0.15, 0.1, 0.075]

    bce_weight_str = str(bce_weights_list[random.randint(0, len(bce_weights_list)-1)]) + ',1'

    weight_decay_list = [0,0,0,0,0,0,0.01]

    weight_decay_str = str(weight_decay_list[random.randint(0, len(weight_decay_list)-1)])


    print("kernel_sizes_str:" + kernel_sizes_str)
    print("number_filter_maps:" + number_filter_maps)
    print("fc_dropout:" + fc_dropout)
    print("embed_dropout:" + embed_dropout)
    print("batch_size:" + batch_size)
    print("bce_weight_str:" + bce_weight_str)
    print("weight_decay: " + weight_decay_str)

    var_list = ["./train.sh",                   # Shell script

                "training/training_conv_encoder.py",        # Training script

                "../../../Data/sorted_sums_matched_struc_train.csv", # Data path

                "../../../Data/vocab.csv",         # Vocab path

                "conv_encoder",                    # Model

                "5",                               # Num epochs

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

                weight_decay_str]

    string = ""

    for idx in range(len(var_list)):

        if idx < (len(var_list) - 1):

            string = string + var_list[idx] + ' '

        else:

            string += var_list[idx]

    subprocess.call(shlex.split(string))
