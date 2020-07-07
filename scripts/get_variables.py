import argparse
from models import *
from utils.build import *
from pruning.prune import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, help='*.cfg path')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--names', type=str, help='*.names path')
    parser.add_argument('--weights', type=str, help='*.weights path')
    parser.add_argument('--imgsz', type=int, default=416, help='image size')
    parser.add_argument('--device', action='store_true', help='device: cpu False or gpu True')
    parser.add_argument('--pool-type', type = str, default='max', help = 'filter representation')

    opt = parser.parse_args()

    # Initialize model
    model = YOLO(opt.cfg, opt.data, opt.names, opt.weights, opt.imgsz, device = False)

    # Extracts all feature maps from all layers of the network for each image in the dataset
    feature_maps, conv_maps, labels = get_feature_maps(model, opt.data, img_size = opt.imgsz, subset = 'train', route = False, debug = -1)

    if len(feature_maps) == len(conv_maps):
        print('Number of images:', len(feature_maps))
    print('Number of feature maps:', len(feature_maps[0]))
    print('Number of convolutional output maps:', len(conv_maps[0]))

    # Represents the output of the filters that compose the network as feature vectors for each image
    X = filter_representation(conv_maps, pool_type = opt.pool_type)
    print('Shape of input variables:', X.shape)

    # Computes the class label matrix of the training data
    Y = class_label_matrix(labels, num_classes = 2)
    print('Shape of output variables:', Y.shape)

    filename = 'variables_' + opt.pool_type + '.npy'
    with open(filename, 'wb') as f:
        np.save(f, X)
        np.save(f, Y)