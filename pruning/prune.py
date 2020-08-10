import torch
import random
import pandas as pd
from tqdm import tqdm
from utils.utils import *
from utils.parse_config import *
from torch.utils.data import DataLoader
from utils.datasets import *
from sklearn.cross_decomposition import PLSRegression

def to_prune(model):

    """ Returns the indexes of the convolutional blocks that can be pruned."""

    blocks = list()

    for i in range(len(model.module_list)):
        try:
            for j in range(len(model.module_list[i])):
                block = model.module_list[i][j]
                next_block = str(model.module_list[i+1]).split('(')[0]
                # It must be a sequential block containing "Conv2d + BatchNorm2d + LeakyReLU" and that does not precede a YOLO layer
                if str(block).split('(')[0] == 'Conv2d' and i+1 not in model.yolo_layers and next_block == 'Sequential': #and len(model.module_list[i+1]) > 1:
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

    # After YOLO Layer
    if block in [layer-3 for layer in model.yolo_layers[:-1]]:

        # Get information from the next convolutional layer
        hyperparameters, parameters = get_layer_info(model.module_list[block+5][0])

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
        model = replace_layer(model, block+5, pruned_conv_layer)


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

def compute_vip(model):

    """ Calculates Variable Importance in Projection (VIP) from PLSRegression model. 
        (https://github.com/scikit-learn/scikit-learn/issues/7050) """

    # Matrices
    W = model.x_weights_
    T = model.x_scores_
    Q = model.y_loadings_

    # Number of features and number of components
    p, c = W.shape
    # Number of observations
    n, _ = T.shape

    # Variable Importance in Projection (VIP)
    VIP = np.zeros((p,))
    S = np.diag(T.T @ T @ Q.T @ Q).reshape(c, -1)
    S_cum = np.sum(S)
    for i in range(p):
        weight = np.array([(W[i,j] / np.linalg.norm(W[:,j]))**2 for j in range(c)])
        VIP[i] = np.sqrt(p*(S.T @ weight)/S_cum)

    return VIP

def pls_vip(model, X, Y, c):

    """ Computes the importance of convolutional filters based on PLS-VIP method.
        Paper: Pruning Deep Networks using Partial Least Squares (https://arxiv.org/pdf/1810.07610.pdf) """

    # Project high dimensional space onto a low dimensional space (latent space)
    PLS = PLSRegression(n_components = c)
    PLS.fit(X.T, Y)

    # Variable Importance in Projection (VIP) for each feature
    VIP = compute_vip(PLS)

    # Filters per layer
    blocks = to_prune(model)
    n_filters = list()
    for block in blocks:
        n_filters.append(int(model.module_list[block][0].out_channels))

    # Importances per layer
    VIPs = list()
    importances = list()

    for block in range(len(blocks)):

        start = sum(n_filters[:block])
        end = sum(n_filters[:block+1])
        VIPs.append(VIP[start:end])

        for filter in range(len(VIPs[block])):
            importances.append([blocks[block], filter, VIPs[block][filter]])

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
    if len(blocks) != len(n_filters):
        raise AssertionError('%d != %d\n' % (len(blocks), len(n_filters)))
    for i in range(len(blocks)):
        selected.append(importances.query('Block == @blocks[@i]')[:n_filters[i]].sort_values(by = 'Filter', ascending = False))
    selected = pd.concat(selected)

    # Returns tuple with less important filters
    return list(selected.to_records(index=False))

def ranked_pruning(model, rate, rank, X = None, Y = None, c = None):

    """ Criteria-based pruning of convolutional filters in the model. """
  
    norms = ['L0', 'L1', 'L2', 'L-INF']

    print('Criteria-based pruning %s\n' % (rank.upper()))

    # Number of filters per layer to be removed
    n_filters = per_layer(model, rate)

    if rank.upper() in norms:
        importances = norm(model, order = rank)
        selected = select_filters(importances, n_filters, ascending = True)
    elif rank.upper() == 'PLS-VIP':
        importances = pls_vip(model, X, Y, c)
        selected = select_filters(importances, n_filters, ascending = True)
    else:
      raise AssertionError('The rank %s does not exist. Try L0, L1, L2, L-Inf or PLS-VIP.' % (rank))

    for i in range(len(selected)):
        block, filter, importance = selected[i]
        model = single_pruning(model, block, filter)

    print('%d filters were pruned.' % (len(selected)))

    return model

def random_pruning(model, rate, seed = 42):

    """ Random pruning of convolutional filters in the model. """

    if seed != -1:
        print('Random pruning with seed %d\n' % (seed))
    else:
        print('Random pruning without seed\n')

    blocks = to_prune(model)

    # Number of filters per layer to be removed
    n_filters = per_layer(model, rate)

    if len(blocks) != len(n_filters):
        raise AssertionError('%d != %d\n' % (len(blocks), len(n_filters)))

    for i in range(len(blocks)):

        if seed != -1:
            random.seed(seed)
        filters = -np.sort(-np.array(random.sample(range(model.module_list[blocks[i]][0].out_channels), n_filters[i])))

        for filter in filters:
            model = single_pruning(model, blocks[i], filter)

    print('%d filters were pruned.' % (sum(n_filters)))

    return model

def feature_extraction(conv_map, pool_type = 'max'):

    """ Represents the output of the filters that compose the network as feature vectors for a single image. """

    if pool_type.lower() == 'avg':
        global_pool = torch.nn.AdaptiveAvgPool2d(output_size = (1, 1))
    else:
        global_pool = torch.nn.AdaptiveMaxPool2d(output_size = (1, 1))

    # Representation of the filters
    features = list()

    # For each convolutional layer
    for l in range(len(conv_map)):

        # For each batch
        for b in range(len(conv_map[l])):
    
            n_filters = len(conv_map[l][b])

            # For each filter
            for f in range(n_filters):

                feature = global_pool(conv_map[l][b][f].unsqueeze(0))
                features.append(float(feature))

    return features

def filter_representation(model, data, img_size, pool_type = 'max', subset = 'train', route = False):

    """ Extract features from all convolutional maps for each image in the subset. """

    # Initializing activation map lists
    inputs = list()
    conv_i = list()
    out_i = list()
    yolo_out_i = list()

    # Initializing the model
    device = torch_utils.select_device()
    model = model.to(device)

    # Prunable block indexes
    blocks = to_prune(model)

    # Load dataset images
    data = parse_data_cfg(data)
    path = data[subset]
    dataset = LoadImagesAndLabels(path = path, img_size = img_size, rect = True, single_cls = False)

    # Get convolutional feature maps for each image
    for i in tqdm(range(len(dataset)), desc = 'Extracting activation maps per image'):

        # Image pre-processing
        img0, _, _ = load_image(dataset, i)
        img = letterbox(img0, new_shape=img_size)[0]

        x = torch.from_numpy(img)
        # Normalization
        x = x.to(device).float() / 255.0
        x = x.float() / 255.0
        x = x.unsqueeze(1)
        x = x.permute(1, 3, 2, 0)

        # For each layer
        for j, module in enumerate(model.module_list):

            name = module.__class__.__name__

            # Sum with WeightedFeatureFusion() and concat with FeatureConcat()
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  
                x = module(x, out_i)
            elif name == 'YOLOLayer':
                yolo_out_i.append(module(x, out_i))
            # Run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc
            else:

                if name == 'Sequential':

                    # Block (Conv2D + BatchNorm2D + LeakyReLU)
                    if len(module) > 1:

                        # Convolution
                        Conv2D = module[0]
                        x = Conv2D(x)
                        if j in blocks:
                           conv_i.append(x)
                        
                        # Batch normalization
                        BatchNorm2D = module[1]
                        x = BatchNorm2D(x)
                        
                        # Activation
                        LeakyReLU = module[2]
                        x = LeakyReLU(x)

                    else:
                        # Single Conv2D
                        x = module(x)
                        if j in blocks:
                            conv_i.append(x)

                # Upsample
                else:
                    x = module(x)

            if route is True:
                out_i.append(x if model.routs[j] else [])
            else:
                out_i.append(x)

        # Feature extraction
        features = feature_extraction(conv_i, pool_type = pool_type)
        inputs.append(features)
            
        # Clearing GPU Memory
        conv_i.clear()
        out_i.clear()
        yolo_out_i.clear()
        del x, img0, img

    return inputs, dataset.labels

def class_label_matrix(labels, num_classes = 2):

    """ Computes the class label matrix of the training data. """

    # Class label matrix
    Y = list()

    for sample in range(len(labels)):
        # False positive sample
        if len(labels[sample]) == 0:
            Y.append(0)
        else:
            Y.append(1)

    # Converts a class vector (integers) to binary class matrix
    if num_classes > 2:
        Y = np.eye(num_classes, dtype = 'uint8')[Y]
    Y = np.array(Y)

    return Y