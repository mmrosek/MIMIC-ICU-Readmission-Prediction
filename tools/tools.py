"""
    Various methods are kept here to keep the other code files simple
"""
import torch
from models import models_mmnet as models

def pick_model(args, dicts, num_struc_feats):
    """
        Use args to initialize the appropriate model
    """
    
    # Preprocessing
    kernel_sizes = [size for size in args.kernel_sizes if size.isnumeric()] # Removing commas if multiple filter sizes passed
    print(kernel_sizes)
    
    if args.bce_weights:
        bce_weights = str(args.bce_weights).split(",")
    else:
        bce_weights = None
        
    if args.struc_dropout_list:
        dropouts = [float(size) for size in args.struc_dropout_list.split(",")]
    else:
        dropouts = []
                
    struc_layers = [int(size) for size in args.struc_layer_size_list.split(",")]
    
    if args.post_merge_layer_size_list:
        post_merge_layers = [int(size) for size in args.post_merge_layer_size_list.split(",")]
    else:
        post_merge_layers = []
        
    print("\nBCE weights: " + str(bce_weights))
    
    if args.model == "conv_encoder":
                
        model = models.ConvEncoder(args.embed_file, kernel_sizes, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.fc_dropout_p, 
                                   args.conv_activation, bce_weights, args.embed_dropout_bool, args.embed_dropout_p, args.loss, args.post_conv_fc_bool)
    
    elif args.model == "mmnet":
        
        model = models.MMNet(args.embed_file, kernel_sizes, args.num_filter_maps, args.gpu, dicts, args.embed_size, 
                             args.fc_dropout_p, args.conv_activation, bce_weights, args.embed_dropout_bool, 
                             args.embed_dropout_p, args.loss, num_struc_feats, struc_layers, dropouts, args.struc_activation, 
                             post_merge_layers, args.post_conv_fc_bool, args.post_conv_fc_dim, args.batch_norm_bool)
        
    elif args.model == "saved":
        model = torch.load(args.test_model)    
    
    print("\nGPU: " + str(args.gpu))

    if args.gpu:
        model.cuda()
                
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.kernel_sizes, args.fc_dropout_p, args.embed_dropout_bool, args.embed_dropout_p, args.num_filter_maps,
        args.command, args.weight_decay, args.data_path, args.embed_file, args.lr]
    
    param_names = ["kernel_sizes", "fc_dropout", "embed_dropout_bool", "embed_dropout_p", "num_filter_maps", "command",
        "weight_decay", "data_path", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params
