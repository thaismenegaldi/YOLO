import torch

def to_prune(model):

    """ Returns the indexes of convolutional blocks in the model."""

    blocks = list()

    for i in range(len(model.module_list)):
        try:
            for j in range(len(model.module_list[i])):
                block = model.module_list[i][j]
                if str(block).split('(')[0] == 'Conv2d' and i+1 not in model.yolo_layers:
                    blocks.append(i)

        except:
            pass

    return blocks