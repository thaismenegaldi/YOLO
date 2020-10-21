import os
import argparse
import subprocess
import numpy as np

def evaluate_model():

    # Opens the temporary file
    f = open('../eval.txt', 'a+')

    # Running evaluation algorithm and saving to temporary file
    subprocess.call('./darknet detector map dfire.data dfire.cfg weights/dfire_best.weights', shell = True, stdout = f)

    # Closing file
    f.close()

def change_set(set):

    # Opens the file in read-only mode
    f = open('dfire.data', 'r')

    # Reads lines until EOF
    lines = f.readlines()

    # Loop over the lines
    for i, line in enumerate(lines):
        if 'valid' in line:
          lines[i] = 'valid = data/dfire_' + set + '.txt\n'

    # Opens the file in write-only mode
    f = open('dfire.data', 'w')

    # Changing validation set in the data file
    f.writelines(lines)
    # Closing file
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--iter', type = str, help = 'Number of iterations')
    parser.add_argument('--method', type = str, help = 'Pruning method')
    opt = parser.parse_args()

    # Open root folder
    root = 'Fine-Tuning/' + opt.iter + os.sep + opt.method + os.sep
    os.chdir(root)

    # Pruned models with pruning rate from 5% to 95%
    folders = np.arange(start = 5, stop = 100, step = 5)

    for folder in folders:

        # Open current folder
        subdir = str(folder) + os.sep
        os.chdir(subdir)

        # Evaluates model in training set
        change_set('train')
        evaluate_model()

        # Evaluates model in validation set
        change_set('valid')
        evaluate_model()

        # Returns to root folder
        os.chdir('../')