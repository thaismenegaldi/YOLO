import torch
import numpy as np
import pruning.prune

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

def summary(model, input_shape = (3, 416, 416), name = 'YOLO'):

    """ Prints a summary representation of your model built by class ModuleList().
        Similar to model.summary() in Keras. """
    
    total_params = 0
    total_layers = 0
    count = 0
    layers = list()
    output_shape = input_shape

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
                  total_params += num_params

                  print('{:>20} {:>20} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))
        
        except:

              count += 1
              layer = str(model.module_list[i]).split('(')[0]
              layers.append(layer)
              
              if layer == 'Upsample':
                  factor = int(model.module_list[i].scale_factor)
                  num_params = 0
                  output_shape = (output_shape[0], factor*output_shape[1], factor*output_shape[2])
                  print('{:>20} {:>20} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))
              
              elif layer == 'WeightedFeatureFusion':
                  num_params = 0
                  print('{:>10} {:>15} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))

              else:
                  num_params = 0
                  print('{:>20} {:>20} {:>15}'.format(layer + '-' + str(count), str(output_shape), '{:,}'.format(num_params)))

              total_params += num_params
            
        if i != len(model.module_list)-1:
            print('---------------------------------------------------------------------')

    assert total_params == sum(x.numel() for x in model.parameters()), 'Parameter count error'

    # Assume 4 bytes/number (float on cuda).
    total_params_size = round(abs(total_params * 4. / (1024 ** 2.)), 2)

    # Number of gradients
    total_grads = sum(grad.numel() for grad in model.parameters() if grad.requires_grad)

    # Set of layers
    layers = {x:layers.count(x) for x in set(layers)}
    
    print('=====================================================================')

    for key, value in layers.items():
        total_layers += value
        print(key + ' layers:', value)

    print('Total number of layers:', total_layers)

    print('---------------------------------------------------------------------')
    print('Parameters:', '{:,}'.format(total_params))
    print('Gradients:', '{:,}'.format(total_grads))
    print('Parameters size (MB):', '{:,}'.format(total_params_size))
    print('---------------------------------------------------------------------')

def view(model, block, filter):

    print('MODEL BEFORE PRUNING\n')
    print(model.module_list[block:block+2], '\n')

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

    model = pruning(model, block, filter)

    print('\nMODEL AFTER PRUNING \n')
    print(model.module_list[block:block+2], '\n')

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