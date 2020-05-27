import torch

def to_prune(model):

    """ Returns the indexes of convolutional layers in the model."""

    layers = list()

    for i in range(len(model.module_list)):
        try:
            for j in range(len(model.module_list[i])):
                layer = model.module_list[i][j]
                if str(layer).split('(')[0] == 'Conv2d':
                    layers.append(i)

        except:
            pass

    return layers