import torch
import pandas as pd
from utils.utils import *
from utils.parse_config import *
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels

def to_prune(model):

    """ Returns the indexes of the convolutional blocks that can be pruned."""

    blocks = list()

    for i in range(len(model.module_list)):
        try:
            for j in range(len(model.module_list[i])):
                block = model.module_list[i][j]
                next_block = str(model.module_list[i+1]).split('(')[0]
                # It must be a sequential block containing "Conv2d + BatchNorm2d + LeakyReLU" and that does not precede a YOLO layer
                if str(block).split('(')[0] == 'Conv2d' and i+1 not in model.yolo_layers and next_block == 'Sequential' and len(model.module_list[i+1]) > 1:
                    blocks.append(i)

        except:
            pass

    return blocks

def prunable_filters(model):

    """ Computes number of prunable filters. """

    n_filters = 0

    blocks = to_prune(model)

    for block in blocks:
        n_filters += model.module_list[block][0].weight.data.shape[0]

    return n_filters

def get_layer_info(layer):

    """ Extracts information that makes up the layer, as well as its weights and bias. """

    hyperparameters = dict()
    parameters = dict()

    # Convolutional layer
    if str(layer).split('(')[0] == 'Conv2d':
        
        hyperparameters['in_channels'] = layer.in_channels
        hyperparameters['out_channels'] = layer.out_channels
        hyperparameters['kernel_size'] = layer.kernel_size
        hyperparameters['stride'] = layer.stride
        hyperparameters['padding'] = layer.padding

        if layer.bias is not None:
            hyperparameters['bias'] = True
            parameters['bias'] = layer.bias.clone()
        else:
            hyperparameters['bias'] = False
            parameters['bias'] = None
        parameters['weight'] = layer.weight.clone()

    # Batch normalization layer
    elif str(layer).split('(')[0] == 'BatchNorm2d':

        hyperparameters['num_features'] = layer.num_features
        hyperparameters['eps'] = layer.eps
        hyperparameters['momentum'] = layer.momentum
        hyperparameters['affine'] = layer.affine
        hyperparameters['track_running_stats'] = layer.track_running_stats

        parameters['bias'] = layer.bias.clone()
        parameters['weight'] = layer.weight.clone()
        
    return hyperparameters, parameters

def replace_layer(model, block, layer):

    """ Replaces original layer with pruned layer. """

    if str(layer).split('(')[0] == 'Conv2d':
        model.module_list[block][0] = layer

    elif str(layer).split('(')[0] == 'BatchNorm2d':
        model.module_list[block][1] = layer

    return model

def remove_filter(parameters, filter, name = 'weight', channels = 'output'):

    """ Removes convolutional filter from a layer. """

    if channels == 'output':

        if filter != 0:

            head_tensor = parameters[name][:filter]
            tail_tensor = parameters[name][filter+1:]
            parameters[name].data = torch.cat((head_tensor, tail_tensor), axis = 0)

        else:
            parameters[name].data = parameters[name][filter+1:]


    elif channels == 'input':

        if filter != 0:

            head_tensor = parameters[name][:,:filter]
            tail_tensor = parameters[name][:,filter+1:]
            parameters[name].data = torch.cat((head_tensor, tail_tensor), axis = 1)

        else:
            parameters[name].data = parameters[name][:,filter+1:]

    return parameters

def single_pruning(model, block, filter):

    """ Pruning a single convolutional filter of the model. """

    # Log file
    log = open('pruned_filters.txt', 'a+')

    # Get information from the current convolutional layer
    hyperparameters, parameters = get_layer_info(model.module_list[block][0])

    # Creates a replica of the convolutional layer to perform pruning
    pruned_conv_layer = torch.nn.Conv2d(in_channels = hyperparameters['in_channels'],
                                        out_channels = hyperparameters['out_channels']-1,
                                        kernel_size = hyperparameters['kernel_size'],
                                        stride = hyperparameters['stride'],
                                        padding = hyperparameters['padding'],
                                        bias = False if parameters['bias'] is None else True                                  
                                        )
    
    # Removes convolutional filter
    parameters = remove_filter(parameters, filter, name = 'weight', channels = 'output')

    # Updates pruned convolutional layer
    pruned_conv_layer.weight.data = parameters['weight'].data
    pruned_conv_layer.weight.requires_grad = True

    if parameters['bias'] is not None:
        parameters = remove_filter(parameters, filter, name = 'bias', channels = 'output')
        pruned_conv_layer.bias.data = parameters['bias'].data
        pruned_conv_layer.bias.requires_grad = True

    # Exchanges the original layer with the pruned layer
    model = replace_layer(model, block, pruned_conv_layer)

    # If the block contains more than one layer, convolutional layer is the first
    if len(model.module_list[block]) > 1:

        # Get information from the current batch normalization layer
        hyperparameters, parameters = get_layer_info(model.module_list[block][1])

        # Creates a replica of the batch normalization layer to perform pruning
        pruned_batchnorm_layer = torch.nn.BatchNorm2d(num_features = hyperparameters['num_features']-1,
                                                      eps = hyperparameters['eps'],
                                                      momentum = hyperparameters['momentum'],
                                                      affine = hyperparameters['affine'],
                                                      track_running_stats = hyperparameters['track_running_stats']
                                                      )
        
        # Removes filter
        parameters = remove_filter(parameters, filter, name = 'weight', channels = 'output')
        parameters = remove_filter(parameters, filter, name = 'bias', channels = 'output') 

        pruned_batchnorm_layer.weight.data = parameters['weight'].data
        pruned_batchnorm_layer.weight.requires_grad = True

        pruned_batchnorm_layer.bias.data = parameters['bias'].data
        pruned_batchnorm_layer.bias.requires_grad = True

        # Exchanges the original layer with the pruned layer
        model = replace_layer(model, block, pruned_batchnorm_layer)

    # If the next block is also sequential
    if str(model.module_list[block+1]).split('(')[0] == 'Sequential':

        # Get information from the next convolutional layer
        hyperparameters, parameters = get_layer_info(model.module_list[block+1][0])

        # Creates a replica of the convolutional layer to perform pruning
        pruned_conv_layer = torch.nn.Conv2d(in_channels = hyperparameters['in_channels']-1,
                                            out_channels = hyperparameters['out_channels'],
                                            kernel_size = hyperparameters['kernel_size'],
                                            stride = hyperparameters['stride'],
                                            padding = hyperparameters['padding'],
                                            bias = False if parameters['bias'] is None else True                                  
                                            )
        
        # Removes convolutional filter
        parameters = remove_filter(parameters, filter, name = 'weight', channels = 'input')

        # Updates pruned convolutional layer
        pruned_conv_layer.weight.data = parameters['weight'].data
        pruned_conv_layer.weight.requires_grad = True

        # Exchanges the original layer with the pruned layer
        model = replace_layer(model, block+1, pruned_conv_layer)

    # Removes convolutional filter from attribute related to .cfg file
    model.module_defs[block]['filters'] -= 1

    # Deletes auxiliary layers
    del pruned_conv_layer
    del pruned_batchnorm_layer

    log.write('Convolutional filter %d pruned from block %d\n' % (filter, block))

    return model

def norm(model, order = 'L2'):

    """ Computes the importance of convolutional filters based on the norm of the weights. """

    if order.upper() == 'L0':
      p = 0
    elif order.upper() == 'L1':
      p = 1
    elif order.upper() == 'L2':
      p = 2
    elif order.upper() == 'L-INF':
      p = float('inf')
    else:
      raise AssertionError('The order %s does not exist. Try L0, L1, L2 or L-Inf.' % (order)) 

    importances = list()

    blocks = to_prune(model)

    for block in blocks:

        n_filters = model.module_list[block][0].weight.data.shape[0]

        for i in range(n_filters):

            importance = model.module_list[block][0].weight[i].data.norm(p).item()
            importances.append([block, i, importance])

    return importances

def per_layer(model, rate):

    """ Calculates the number of filters that will be removed in each layer. """

    n_filters = list()

    blocks = to_prune(model)
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels*rate))
        
    return n_filters

def select_filters(importances, n_filters, ascending = True):

    """ Select the filters to be removed based on their respective importance. """

    importances = pd.DataFrame(importances, columns = ['Block', 'Filter', 'Importance'])
    # Sorting importances
    importances = importances.sort_values(by = 'Importance', ascending = ascending)    

    selected = list()
    # Selecting the filters for each layer that will be pruned
    blocks = list(importances['Block'].drop_duplicates().sort_values(ascending = True))
    print(len(blocks))
    if len(blocks) != len(n_filters):
        raise AssertionError('%d != %d\n' % (len(blocks), len(n_filters)))
    for i in range(len(blocks)):
        selected.append(importances.query('Block == @blocks[@i]')[:n_filters[i]].sort_values(by = 'Filter', ascending = False))
    selected = pd.concat(selected)

    # Returns tuple with less important filters
    return list(selected.to_records(index=False))

def ranked_pruning(model, rate, rank):

    """ Criteria-based pruning of convolutional filters in the model. """
  
    norms = ['L0', 'L1', 'L2', 'L-INF']

    print('Criteria-based pruning %s\n' % (rank.upper()))

    # Number of filters per layer to be removed
    n_filters = per_layer(model, rate)

    if rank.upper() in norms:
        importances = norm(model, order = rank)
        selected = select_filters(importances, n_filters, ascending = True)
    else:
      raise AssertionError('The rank %s does not exist. Try L0, L1, L2 or L-Inf.' % (rank))

    for i in range(len(selected)):
        block, filter, importance = selected[i]
        model = single_pruning(model, block, filter)

    print('%d filters were pruned.' % (len(selected)))

    return model

def random_pruning(model, rate):

    """ Random pruning of convolutional filters in the model. """

    blocks = to_prune(model)

    n_filters = int(prunable_filters(model) * rate)

    for n in range(n_filters):

        block = blocks[np.random.randint(low = 0, high = len(blocks), size = 1)[0]]
        filter = np.random.randint(low = 0, high = model.module_list[block][0].out_channels, size = 1)[0]

        model = single_pruning(model, block, filter)

    return model