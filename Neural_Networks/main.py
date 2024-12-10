# import modules
from utils.utils import create_directory
from utils.utils import read_dataset

import os
import numpy as np
import sklearn
import sys

from utils import utils

from utils.utils import viz_cam
from utils.utils import multi_dimensional_scaling
from utils.utils import visualize_filter
from utils.utils import plot_conv
from utils.utils import occlusion


def fit_classifier(output_directory):

    # import data
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    # number of different classes
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save original y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    # if uni-variate
    if len(x_train.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # dimensions
    input_shape = x_train.shape[1:]

    # create classifier
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    # training
    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):

    # add here different types of classifiers
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)


# main

# change this directory for your machine
root_dir = '/Users/leluwy/Desktop/ETH/Bachelorprojekt/Multivariate_arff'

# this is the code used to launch an experiment on a dataset
archive_name = 'EyeMovements'
dataset_name = 'EyeMovements'
classifier_name = 'cnn'
itr = ''

output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                    dataset_name + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', archive_name, dataset_name, classifier_name, itr)

if os.path.exists(test_dir_df_metrics):
    # viz_cam(root_dir, archive_name, dataset_name, classifier_name, itr, 0)
    # visualize_filter(root_dir, archive_name, dataset_name, classifier_name, itr, 6)
    # plot_conv(root_dir, archive_name, dataset_name, classifier_name, itr, 0)
    # multi_dimensional_scaling(root_dir, archive_name, dataset_name, classifier_name, itr, dimension=0)
    # occlusion(root_dir, archive_name, dataset_name, classifier_name, itr)
    print('Already done')

else:

    create_directory(output_directory)
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    fit_classifier(output_directory)

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
