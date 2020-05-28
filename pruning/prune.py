import torch

def to_prune(model):

    """ Returns the indexes of convolutional blocks in the model."""

    blocks = list()

    for i in range(len(model.module_list)):
        try:
            for j in range(len(model.module_list[i])):
                block = model.module_list[i][j]
                next_block = str(model.module_list[i+1]).split('(')[0]
                if str(block).split('(')[0] == 'Conv2d' and i+1 not in model.yolo_layers and next_block == 'Sequential':
                    blocks.append(i)

        except:
            pass

    return blocks

def get_layer_info(layer):

    hyperparameters = dict()
    parameters = dict()

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

    if str(layer).split('(')[0] == 'Conv2d':
        model.module_list[block][0] = layer

    elif str(layer).split('(')[0] == 'BatchNorm2d':
        model.module_list[block][1] = layer

    return model

def remove_filter(parameters, filter, name = 'weight', channels = 'output'):

    if channels == 'output':

        head_tensor = parameters[name][:filter-1]
        tail_tensor = parameters[name][filter:]
        parameters[name].data = torch.cat((head_tensor, tail_tensor), axis = 0)

    elif channels == 'input':

        head_tensor = parameters[name][:,:filter-1]
        tail_tensor = parameters[name][:,filter:]
        parameters[name].data = torch.cat((head_tensor, tail_tensor), axis = 1)

    return parameters