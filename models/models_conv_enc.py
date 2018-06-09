import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
from torch.optim import Adam
from dataproc import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, embed_file, dicts, embed_size=100):
        super(BaseModel, self).__init__()
        self.embed_size = embed_size

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
            print("Size of embedding matrix")
            print(W.size())
            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
            self.embed.weight.requires_grad = True # Likely not needed

        else:
            vocab_size = len(dicts[0])
            self.embed = nn.Embedding(vocab_size+2, embed_size) #add 2 to include UNK and PAD


    def weighted_bce(self, y_pred, target, weights=None):

        '''Computes weighted binary cross entropy given predictions and ground truths (target).
           weights: [weight_neg_class, weight_pos_class]'''
        
        if weights is not None:
            assert len(weights) == 2
            
            loss = float(weights[1]) * (target * torch.log(y_pred)) + \
                   float(weights[0]) * ((1 - target) * torch.log(1 - y_pred))
                   
        else:
            
#            loss = F.binary_cross_entropy(y_pred, target)
            loss = target * torch.log(y_pred) + (1 - target) * torch.log(1 - y_pred)
    
        return torch.neg(torch.mean(loss))
    
    
    def margin_ranking_loss(self, preds, labels, margin):
    
        batch_size = list(preds.size())[0]
        
        sum_errors = Variable(torch.FloatTensor(1,1).zero_(), requires_grad = True)
        
        for i in range(batch_size):
        
            for j in range(i+1, batch_size):
                
                # if preds[i] > preds[j], labels[i] should be bigger than labels[j]
                pred_diff = preds[i] - preds[j] 
        
                label_diff = labels[i] - labels[j] 
                        
                if (pred_diff * label_diff).data[0] < margin: # if preds[i] not > preds[j] by margin, then there is a penalty
                
                    sum_errors = sum_errors - (pred_diff * label_diff) + margin
                                   
        sum_errors = sum_errors/batch_size
        
        return sum_errors


    def params_to_optimize(self):
        return self.parameters()
        
        
        
class ConvEncoder(BaseModel):

    def __init__(self, embed_file, kernel_sizes, num_filter_maps, gpu=True, dicts=None, embed_size=100, fc_dropout_p=0.5, conv_activation = "selu", bce_weights=None, embed_dropout_bool = False, embed_dropout_p = 0.2, loss = "BCE"):
        super(ConvEncoder, self).__init__(embed_file, dicts) 
           
        self.bce_weights = bce_weights
        self.embed_dropout_bool = embed_dropout_bool
        self.loss_fx = loss

        # Setting non-linear activation on feature maps
        self.conv_activation = getattr(F, conv_activation) # Equivalent to F.[conv_activation]
        print("\nConv Activation: " + str(self.conv_activation))
        
        # Initializing convolutional layers
        self.conv_layers = [nn.Conv1d(in_channels=embed_size, out_channels=num_filter_maps, 
                            kernel_size=int(kernel_size)) for kernel_size in kernel_sizes]
        
        for i, conv_layer in enumerate(self.conv_layers):
            self.add_module('conv_%d' % i, conv_layer) # Add convolutional modules, one for each kernel size
            
        # dropout
        self.embed_dropout = nn.Dropout(p = embed_dropout_p)
        self.fc_dropout = nn.Dropout(p = fc_dropout_p)

        # fully-connected layer         
        maxpool_output_dim = num_filter_maps * len(kernel_sizes)
        self.fc = nn.Linear(maxpool_output_dim, 1) 
        xavier_uniform(self.fc.weight)

    def forward(self, x, target):
        
        #embed
        x = self.embed(x)
        
        if self.embed_dropout_bool:
            x = self.embed_dropout(x)

        x = x.transpose(1, 2) # Transposing from word vectors as rows --> word vectors as columns  
                  
        # Aggregating outputs for each filter for each kernel size
        filter_outputs = []
        for i in range(len(self.conv_layers)):
            conv_layer = getattr(self, 'conv_{}'.format(i)) # Equivalent to self.conv_i
            filter_outputs.append(
                    self.conv_activation(conv_layer(x)).max(dim=2)[0]) # .max() returns 2 arrays; 0. max vals, 1. argmax (max indices across desired dimension --> dim=2 is across columns in 3d tensor)
                
        
        x = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0] # Concatenating filter outputs into 1d vector

        #linear output
        x = self.fc_dropout(x) # Multiplying by 2 when dropout = 0.5? Should test
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        y = yhat.squeeze()
        
#        print("\nLoss_fx = " + str(self.loss_fx))
        
        if self.loss_fx == "margin_ranking_loss":
            loss = self.margin_ranking_loss(y, target, 0.15) # Need to add parameter for margin
        else:
            loss = self.weighted_bce(y,target, self.bce_weights)
        
        return y, loss



































