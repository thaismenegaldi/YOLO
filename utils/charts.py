import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scheduler_plot(optimizers, schedulers, config, epochs):

    """ Visualize the learning rate schedule. """

    assert (len(optimizers) == len(schedulers)), "For each optimizer, there is a scheduler"

    plt.figure(figsize = config['figsize'])
    for i in range(len(optimizers)):

        y = []
        for _ in range(epochs):
            optimizers[i].step()
            schedulers[i].step()
            y.append(optimizers[i].param_groups[0]['lr'])
        
        plt.plot(y, config['linestyle'], label = config['label'], color = config['color'][i])

    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    plt.title(config['title'])
    if config['bbox_to_anchor'] != None:
        plt.legend(config['legend'], loc = config['loc'], bbox_to_anchor = config['bbox_to_anchor'])
    else:
        plt.legend(config['legend'], loc = config['loc'])
    if config['grid'] is True:
        plt.grid()
    plt.tight_layout()

    if config['save'] is True:
        plt.savefig(config['filename'], dpi = config['dpi'])

def learning_curve(data, config):

    """ Visualize the learning curve. 
        It is recommended when it is necessary to check the relationship between the size of the training set and the performance of the model. """

    assert (len(data.columns) == 10), "Make sure the dataframe is in the desired format"

    train = {
        'mAP': data.iloc[:, 2],
        'AP smoke': data.iloc[:, 3],
        'AP fire': data.iloc[:, 4],
        'F1-Score': data.iloc[:, 5]
    }

    valid = {
        'mAP': data.iloc[:, 6],
        'AP smoke': data.iloc[:, 7],
        'AP fire': data.iloc[:, 8],
        'F1-Score': data.iloc[:, 9]
    }

    plt.figure(figsize = config['figsize'])

    metrics = list(train.keys())

    for i in range(len(train)):
        
        plt.subplot(2, 2, i+1)
        plt.plot(data.iloc[:, 1], train[metrics[i]], config['linestyle'], color = config['color'][0])
        plt.plot(data.iloc[:, 1], valid[metrics[i]], config['linestyle'], color = config['color'][1])
        plt.legend(config['legend'])
        plt.xlabel('Training images')
        plt.ylabel(metrics[i])

        if config['grid'] is True:
            plt.grid()

    plt.tight_layout(pad = config['pad'])

    if config['save'] is True:
        plt.savefig(config['filename'], dpi = config['dpi'])