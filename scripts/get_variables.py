import argparse
from models import *
from utils.build import *
from pruning.prune import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='*.cfg path')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--names', type=str, help='*.names path')
    parser.add_argument('--weights', type=str, help='*.weights path')
    parser.add_argument('--imgsz', type=int, default=416, help='image size')
    parser.add_argument('--pool-type', type = str, default='max', help = 'pooling operation')

    opt = parser.parse_args()

    # Output filename
    filename = 'variables_' + opt.pool_type + '.npy'

    # Initialize model
    model = YOLO(opt.cfg, opt.data, opt.names, opt.weights, opt.imgsz)

    # Extracts all feature maps from all layers of the network for each image in the dataset
    inputs, labels = filter_representation(model, opt.data, img_size = opt.imgsz, pool_type = opt.pool_type)

    # Reshape input variables (filters x images)
    X = np.array(inputs).reshape((len(inputs[0]), len(inputs)))

    # Saving matrix X
    with open(filename, 'wb') as f:
        np.save(f, X)

    print('Number of images:', X.shape[1])
    print('Number of filters per image:', X.shape[0])

    print('Shape of input variables:', X.shape)

    # Computes the class label matrix of the training data
    Y = class_label_matrix(labels, num_classes = 2)
    print('Shape of output variables:', Y.shape)

    # Saving matrix Y
    with open(filename, 'wb') as f:
        np.save(f, Y)