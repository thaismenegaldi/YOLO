import torch
import numpy as np
from pruning.prune import *

def sparsity(model, describe = False):

    """ Calculates the sparsity of the convolutional layers, that is, the number of null filters in each layer and in the entire model. """

    sparsities = list()
    idx = 0

    for module in model.module_list:
        try:
            for layer in module:
                idx += 1
                if str(layer).split('(')[0] == 'Conv2d':
                    sparsity = 100. * float(torch.sum(layer.weight == 0)) / float(layer.weight.nelement())
                    sparsities.append(sparsity)
                    if describe is True:
                        print('Sparsity Conv2d-%s: %.2f%%' % (str(idx), sparsity))
        except:
            idx += 1
            pass
    
    print('Global sparsity: %.2f%%' % np.mean(np.array(sparsities)))

def summary(model, input_shape = (3, 416, 416), name = 'YOLO', verbose = True):

    """ Prints a summary representation of your model built by class ModuleList().
        Similar to model.summary() in Keras. """
    
    total_layers = 0
    count = 0
    layers = list()
    output_shape = input_shape

    if verbose:

        print('{:>42}'.format(name + ' Model Summary'))
        print('---------------------------------------------------------------------')
        print('{:>20} {:>20} {:>20}'.format('Layer (Type)', 'Output Shape', 'Parameters #'))
        print('=====================================================================')

    for i in range(len(model.module_list)):

        try:

            for j in range(len(model.module_list[i])):

                layer = str(type(model.module_list[i][j])).split('.')[-1].split("'")[0]
                layers.append(layer)

                if layer == 'Conv2d':
                
                    in_channels = model.module_list[i][j].in_channels
                    out_channels = model.module_list[i][j].out_channels
                    kernel_size = model.module_list[i][j].kernel_size[0]
                    padding = model.module_list[i][j].padding[0]
                    stride = model.module_list[i][j].stride[0]

                    out_size = (output_shape[1] + 2*padding - kernel_size)/stride + 1
                    output_shape = (int(out_channels), int(out_size), int(out_size))

                    if str(model.module_list[i][j].bias) == 'None':
                        num_params = np.power(kernel_size, 2) * in_channels * out_channels
                    else:
                        num_params = np.power(kernel_size, 2) * in_channels * out_channels + out_channels        

                elif layer == 'BatchNorm2d':
                    num_features = model.module_list[i][j].num_features
                    num_params = 2*num_features

                else:
                    num_params = 0


                count += 1

                if verbose:
                    print('{:>20} {:>20} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))
        
        except:

            count += 1
            layer = str(model.module_list[i]).split('(')[0]
            layers.append(layer)
            
            if layer == 'Upsample':
                factor = int(model.module_list[i].scale_factor)
                num_params = 0
                output_shape = (output_shape[0], factor*output_shape[1], factor*output_shape[2])
                if verbose:
                    print('{:>20} {:>20} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))
            
            elif layer == 'WeightedFeatureFusion':
                num_params = 0
                if verbose:
                    print('{:>10} {:>15} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))

            else:
                num_params = 0
                if verbose:
                    print('{:>20} {:>20} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))
           
        if i != len(model.module_list)-1:
            if verbose:
                print('---------------------------------------------------------------------')

    # Number of parameters
    total_params = sum(x.numel() for x in model.parameters())

    # Assume 4 bytes/number (float on cuda).
    total_params_size = round(abs(total_params * 4. / (1024 ** 2.)), 2)

    # Number of gradients
    total_grads = sum(grad.numel() for grad in model.parameters() if grad.requires_grad)

    # Set of layers
    layers = {x:layers.count(x) for x in set(layers)}
    
    if verbose:
        print('=====================================================================')
    else:
        print('---------------------------------------------------------------------')

    for key, value in layers.items():
        total_layers += value
        print(key + ' layers:', value)

    print('Total number of layers:', total_layers)

    print('---------------------------------------------------------------------')
    print('Parameters:', '{:,}'.format(total_params))
    print('Gradients:', '{:,}'.format(total_grads))
    print('Parameters size (MB):', '{:,}'.format(total_params_size))
    print('---------------------------------------------------------------------')

def view(model, block, filter, verbose = False):

    """ Method for viewing specific blocks of the network. """

    if verbose:

        print('Model block:\n', model.module_list[block:block+2], '\n')
        print('.cfg block:\n', model.module_defs[block], '\n')

    print('Current Conv2d Weights:', model.module_list[block][0].weight.data.shape[0], 'x',
                                        model.module_list[block][0].weight.data.shape[1], 'x',
                                        model.module_list[block][0].weight.data.shape[2], 'x',
                                        model.module_list[block][0].weight.data.shape[3])
    try:
        print('Current Conv2d Bias:', model.module_list[block][0].bias.data.shape[0])
    except:
        pass
    try:
        print('Current BatchNorm2d Weights:', model.module_list[block][1].weight.data.shape[0])
        print('Current BatchNorm2d Bias:', model.module_list[block][1].bias.data.shape[0])
    except:
        pass
    try:
        print('Next Conv2d Weights:', model.module_list[block+1][0].weight.data.shape[0], 'x',
                                        model.module_list[block+1][0].weight.data.shape[1], 'x',
                                        model.module_list[block+1][0].weight.data.shape[2], 'x',
                                        model.module_list[block+1][0].weight.data.shape[3])
        print('Next Conv2d Bias:', model.module_list[block+1][0].bias.data.shape[0])
    except:
        pass

def model_to_cfg(model, 
                 version = 3,
                 cfg = 'dfire.cfg',
                 mode = 'train',
                 img_size = (416, 416, 3), 
                 learning_rate = 0.001,
                 batch = 64,
                 subdivisions = 16,
                 max_batches = 8000,
                 burn_in = 1000,
                 policy = 'steps',
                 steps = [3600, 4200],
                 scales = [.1, .1],
                 momentum = 0.9,
                 decay = 0.0005,
                 angle = 0,
                 saturation = 1.5,
                 exposure = 1.5,
                 hue = .1,
                 mosaic = 1,
                 max_delta = -1):

    """ Converts the PyTorch model to a Darknet .cfg file. """

    file = open(cfg, 'w+')  

    file.write('[net]\n')

    if mode == 'train':

        file.write('# Testing\n')
        file.write('#batch = 1\n')
        file.write('#subdivisions = 1\n')

        file.write('# Training\n')
        file.write('batch = %d\n' % (batch))
        file.write('subdivisions = %d\n' % (subdivisions))

    else: 

        file.write('# Testing\n')
        file.write('batch = 1\n')
        file.write('subdivisions = 1\n')

        file.write('# Training\n')
        file.write('#batch = %d\n' % (batch))
        file.write('#subdivisions = %d\n' % (subdivisions))

    file.write('width = %d\n' % (img_size[0]))
    file.write('height = %d\n' % (img_size[1]))
    
    try:
        file.write('channels = %d\n' % (img_size[2]))
    except:
        file.write('channels = 1\n')

    file.write('momentum = %s\n' % (str(momentum)))
    file.write('decay = %s\n' % (str(decay)))
    file.write('angle = %s\n' % (str(angle)))
    file.write('saturation = %s\n' % (str(saturation)))
    file.write('exposure = %s\n' % (str(exposure)))
    file.write('hue = %s\n' % (str(hue)))
    file.write('mosaic = %d\n' % (mosaic))
    file.write('\n')

    file.write('learning_rate = %s\n' % str((learning_rate)))
    file.write('burn_in = %d\n' % (burn_in))
    file.write('max_batches = %d\n' % (max_batches))
    file.write('policy = %s\n' % (policy))

    steps_ = ','.join([str(step) for step in steps])    
    file.write('steps = %s\n' % (steps_))
    scales_ = ','.join([str(scale) for scale in scales])    
    file.write('scales = %s\n' % (scales_))
    file.write('\n')

    for block in model.module_defs:

        if block['type'] == 'convolutional':

            if block['stride'] > 1:
                file.write('# Downsample\n\n')

            file.write('[%s]\n' % (block['type']))
            
            if block['batch_normalize'] > 0:

                file.write('batch_normalize = %d\n' % (block['batch_normalize']))
                file.write('filters = %d\n' % (block['filters']))
                file.write('size = %d\n' % (block['size']))
                file.write('stride = %d\n' % (block['stride']))
                file.write('pad = %d\n' % (block['pad']))
                file.write('activation = %s\n\n' % (block['activation']))

            else:

                file.write('size = %d\n' % (block['size']))
                file.write('stride = %d\n' % (block['stride']))
                file.write('pad = %d\n' % (block['pad']))
                file.write('filters = %d\n' % (block['filters']))
                file.write('activation = %s\n\n' % (block['activation']))

        elif block['type'] == 'maxpool':
            
            file.write('[%s]\n' % (block['type']))
            file.write('size = %d\n' % (block['size']))
            file.write('stride = %d\n\n' % (block['stride']))

        elif block['type'] == 'shortcut':

            file.write('[%s]\n' % (block['type']))
            from_ = ','.join([str(fromm) for fromm in block['from']])
            file.write('from = %s\n' % (from_))
            file.write('activation = %s\n\n' % (block['activation']))

        elif block['type'] == 'route':

            file.write('[%s]\n' % (block['type']))
            layers = ','.join([str(layer) for layer in block['layers']])
            file.write('layers = %s\n\n' % (layers))

        elif block['type'] == 'yolo':

            file.write('[%s]\n' % (block['type']))
            masks = ','.join([str(mask) for mask in block['mask']])
            file.write('mask = %s\n' % (masks))

            anchors_ = ', '.join([(str(int(anchor[0])) + ',' + str(int(anchor[1]))) for anchor in block['anchors']])
            file.write('anchors = %s\n' % (anchors_))

            file.write('classes = %d\n' % (block['classes']))
            file.write('num = %d\n' % (block['num']))
            file.write('jitter = %s\n' % (block['jitter']))
            file.write('ignore_thresh = %s\n' % (block['ignore_thresh']))
            file.write('truth_thresh = %d\n' % (block['truth_thresh']))

            # YOLOv3 and variations
            if version == 3:
                file.write('random = %d\n\n' % (block['random']))

            # YOLOv4 and variations
            elif version == 4:
                # Only on the last layer
                try:
                    file.write('random = %d\n' % (block['random']))
                except:
                    pass
                file.write('scale_x_y = %s\n' % (block['scale_x_y']))
                file.write('iou_thresh = %s\n' % (block['iou_thresh']))
                file.write('cls_normalizer = %s\n' % (block['cls_normalizer']))
                file.write('iou_normalizer = %s\n' % (block['iou_normalizer']))
                file.write('iou_loss = %s\n' % (block['iou_loss']))
                file.write('nms_kind = %s\n' % (block['nms_kind']))
                file.write('beta_nms = %s\n' % (block['beta_nms']))
                if max_delta != -1:
                    file.write('max_delta = %d\n\n' % (max_delta))
                else:
                    file.write('\n')

            # Other models
            else:
                print('This model is currently not supported. Try version = 3 or version = 4.')

        elif block['type'] == 'upsample':

            file.write('[%s]\n' % (block['type']))
            file.write('stride = %d\n\n' % (block['stride']))

        else:
          
            print('The block %s was not registered.\n' % (block['type']))

    file.close()