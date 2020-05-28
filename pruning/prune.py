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