import subprocess
import shlex
import random
import math


#Randomizing number of kernels and kernel sizes
for i in range(0, 300):
    fc_layer_sizes_str = ""
    embed_dropout_list_str = ""
    activation_layer = ""
    print("=======================Iteration: "+str(i) + " ===============================")
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

    #Randomizing embed dropout
    embed_dropout_list = []
    for i in range(1, len(fc_layer_sizes)):
        embed_dropout_list.append(random.uniform(0, 0.5))
    embed_dropout_list_str = ','.join(map(str, embed_dropout_list))

    #Randomizing bce weights
    bce_weights_list = [0.33, 0.33, 0.33, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.17, 0.17, 0.143, 0.125]
    bce_weight_str = str(bce_weights_list[random.randint(0, len(bce_weights_list)-1)]) + ',1'

    print("fc_layer_sizes_str:" + fc_layer_sizes_str)
    print("activation_layer:" + activation_layer)
    print("embed_dropout_list_str:" + embed_dropout_list_str)

    var_list = []

    if embed_dropout_list_str == '':
        var_list = ["./train.sh",
                "structured/struc_net.py",
                "../../../Data/struc_data_reversed.svmlight",
                "../../../Models/",
                "4",
                "--gpu",
                "--fc-layer-size-list",
                fc_layer_sizes_str,
                "--fc-activation",
                activation_layer,
                "--batch-size",
                "8",
                "--bce-weights",
                bce_weight_str]
    else:
        var_list = ["./train.sh",
                "structured/struc_net.py",
                "../../../Data/struc_data_reversed.svmlight",
                "../../../Models/",
                "4",
                "--gpu",
                "--fc-layer-size-list",
                fc_layer_sizes_str,
                "--fc-activation",
                activation_layer,
                "--dropout-list",
                embed_dropout_list_str,
                "--batch-size",
                "8",
                "--bce-weights",
                bce_weight_str]

    string = ""

    for idx in range(len(var_list)):

        if idx < (len(var_list) - 1):

            string = string + var_list[idx] + ' '

        else:

            string += var_list[idx]


    subprocess.call(shlex.split(string))
