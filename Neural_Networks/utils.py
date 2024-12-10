from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split

from scipy.interpolate import interp1d
from scipy.io import arff

matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def create_directory(directory_path):
    # create directory
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    # create path
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_dataset(root_dir, archive_name, dataset_name):
    from sklearn import preprocessing
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'Multivariate':

        # multi-variate case
        file_name = cur_root_dir + '/' + dataset_name + '/'

        # read train data
        df_train1 = arff.loadarff(file_name + dataset_name + 'Dimension1_TRAIN.arff')

        df_train_all = arff.loadarff(file_name + dataset_name + '_TRAIN.arff')

        # dimension of the data
        dimension = len(df_train_all[0][0][0])

        # length of the time series
        time_series_length = len(df_train_all[0][0][0][0])

        # number of train cases
        number_of_train_cases = len(df_train_all[0])

        df_x_train = np.zeros((number_of_train_cases, time_series_length, dimension))

        for l in range(number_of_train_cases):
            for k in range(dimension):
                for i in range(time_series_length):
                    df_x_train[l][i][k] = df_train_all[0][l][0][k][i]

        data_train_y = pd.DataFrame(df_train1[0])

        # read test data
        df_test1 = arff.loadarff(file_name + dataset_name + 'Dimension1_TEST.arff')
        data_test_y = pd.DataFrame(df_test1[0])

        df_test_all = arff.loadarff(file_name + dataset_name + '_TEST.arff')

        # number of test cases
        number_of_test_cases = len(df_test_all[0])

        df_x_test = np.zeros((number_of_test_cases, time_series_length, dimension))

        for l in range(number_of_test_cases):
            for k in range(dimension):
                for i in range(time_series_length):
                    df_x_test[l][i][k] = df_test_all[0][l][0][k][i]

        # extract labels
        y_train = (data_train_y.values[:, -1])
        y_test = (data_test_y.values[:, -1])

        # encode labels to integer numbers
        label_encoder = preprocessing.LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        # normalization
        std_ = df_x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (df_x_train - df_x_train.mean(axis=1, keepdims=True)) / std_

        std_ = df_x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (df_x_test - df_x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())

    elif archive_name == 'UCRArchive_2018':

        # uni-variate case
        root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'

        # read train data
        df_train = arff.loadarff(root_dir_dataset + dataset_name + '_TRAIN.arff')
        data_train = pd.DataFrame(df_train[0])
        x_train = data_train.drop(['target'], axis=1)

        # read test data
        df_test = arff.loadarff(root_dir_dataset + dataset_name + '_TEST.arff')
        data_test = pd.DataFrame(df_test[0])
        x_test = data_test.drop(['target'], axis=1)

        # extract labels
        y_train = (data_train.values[:, -1])
        y_test = (data_test.values[:, -1])

        # encode labels to integer numbers
        label_encoder = preprocessing.LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        # to array
        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # normalization
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())

    elif archive_name == 'EyeMovements':

        # multi-variate eye-movement dataset
        file_name = cur_root_dir + '/' + dataset_name + '/'
        X = np.load(file_name + 'X_train.npy')
        y = np.load(file_name + 'y_train.npy')

        print('number of samples in total:', len(X))

        # use only specifc EEG channels (used correlation plots)
        X = X[:, :, :]

        print('input shape:', X.shape)

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=True)

        # standardization
        std_ = X_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (X_train - X_train.mean(axis=1, keepdims=True)) / std_

        std_ = X_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (X_test - X_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())

    else:
        print("no archive found")

    return datasets_dict


def plot_conv(root_dir, archive_name, dataset_name, classifier, itr, dimension):

    # import modules
    from tensorflow import keras

    # read dataset
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    # if uni-variate
    if len(x_train.shape) == 2:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # load model
    model = keras.models.load_model(
        root_dir + '/results/' + classifier + '/' + archive_name + itr + '/' + dataset_name + '/best_model.hdf5')

    # get filter weights from the first layer
    layer = 1
    filters = model.layers[layer].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255), (18 / 255, 80 / 255, 150 / 255),
              (120 / 255, 150 / 255, 1 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255), (230 / 255, 90 / 255, 0 / 255),
                   (0 / 255, 200 / 255, 100 / 255), (230 / 255, 250 / 255, 50 / 255)]

    # which sample to plot
    idx = 470  # 250

    # which filter to plot
    idx_filter = 0

    print('filters shape', filters.shape)

    # plot convolutions
    convolved_filter_1 = new_feed_forward([x_train])[0]

    clas = int(y_train[idx])-1
    print('clas:', y_train[idx])

    if len(x_train.shape) == 2:  # if univariate
        plt.plot(x_train[idx], color=colors[clas], label='class' + str(clas+1) + '-raw')
    else:
        plt.plot(x_train[idx, :, dimension], color=colors[clas], label='class' + str(clas+1) + '-raw')

    conv = np.convolve(x_train[idx, :, dimension], filters[:, dimension, idx_filter])/(0.01*len(x_train[idx, :, dimension]))  # convolution
    plt.plot(conv, color=colors_conv[clas], label='class' + str(clas+1) + '-conv for this specific channel')
    plt.plot(convolved_filter_1[idx, :, dimension], color=colors_conv[clas+1], label='class' + str(clas + 1) + '-conv over all channels')

    plt.plot(filters[:, dimension, idx_filter], label='filter')
    plt.legend()

    # save result
    plt.savefig(root_dir + '-convolution-' + dataset_name + '-' + classifier + str(idx_filter) + '-' + str(dimension) +
                '-' + str(idx) + '.pdf')
    plt.close()

    return 1


def visualize_filter(root_dir, archive_name, dataset_name, classifier, itr, dimension=0):

    # import modules
    from tensorflow import keras

    # load model
    model = keras.models.load_model(
        root_dir + '/results/' + classifier + '/' + archive_name + itr + '/' + dataset_name + '/best_model.hdf5')

    # get filter weights from the first layer
    layer = 1
    filters = model.layers[layer].get_weights()[0]

    print('filters shape', filters.shape)

    # plot filters
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=False)  # works for CNN

    i = 0
    for ax in axs.flat:
        ax.plot(filters[:, dimension, i], color='b')
        # ax.axes.yaxis.set_ticklabels([])
        i = i+1

    # save result
    plt.savefig(root_dir + '-convolution-' + dataset_name + '-' + classifier + '-' + str(dimension) + '.pdf')
    plt.close()

    return 1


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    #  the basis for this function was taken from https: // github.com / hfawaz / dl - 4 - tsc.
    # we adapted it for our specific scenario
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float64), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    #  the basis for this function was taken from https: // github.com / hfawaz / dl - 4 - tsc.
    # we adapted it for our specific scenario
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float64), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def plot_epochs_metric(hist, file_name, metric='loss'):
    #  the basis for this function was taken from https: // github.com / hfawaz / dl - 4 - tsc.
    # we adapted it for our specific scenario
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    #  the basis for this function was taken from https: // github.com / hfawaz / dl - 4 - tsc.
    # we adapted it for our specific scenario
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float64), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def multi_dimensional_scaling(root_dir, archive_name, dataset_name, classifier, itr, dimension=0):

    # import necessary modules
    from tensorflow import keras
    from keras.models import Model

    # read data
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    # mds for raw data
    mds = MDS(2, metric=True, normalized_stress='auto', random_state=42)

    # uni-variate
    if len(x_test.shape) == 2:
        pts = mds.fit_transform(x_test)

    # multi-variate with dimension d
    else:
        pts = mds.fit_transform(x_test[:, :, dimension])

    # get number of different classes
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    colors = ['red', 'blue', 'green', 'orange', 'yellow']
    marker = ['.', 'x', '*', 'p', 'x']

    # plotting for the raw data
    for i in range(nb_classes):
        subset = pts[y_test == i]
        plt.scatter(subset[:, 1], subset[:, 0], color=colors[i], marker=marker[i])

    plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + dataset_name + '-mds-raw' + str(dimension) + '.png', bbox_inches='tight', dpi=1080)
    plt.close()

    # output layer - GAP
    model = keras.models.load_model(root_dir + '/results/' + classifier + '/' + archive_name + itr + '/' + dataset_name
                                    + '/best_model.hdf5')

    # works for fcn and resnet
    layer_name = 'global_average_pooling1d'

    # get the model from the GAP layer
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    GAP_output = intermediate_layer_model.predict(x_test)

    # multi-dimensional-scaling for the GAP layer
    GAP_mds = mds.fit_transform(GAP_output)

    # plotting output of the GAP layer
    for i in range(nb_classes):
        subset = GAP_mds[y_test == i]
        plt.scatter(subset[:, 0], subset[:, 1], color=colors[i], marker=marker[i])

    # save results
    plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + dataset_name + '-mds-GAP' + '.png', bbox_inches='tight', dpi=1080)
    plt.close()


def occlusion(root_dir, archive_name, dataset_name, classifier, itr):

    # import modules
    from tensorflow import keras

    # read data
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    # load model
    model = keras.models.load_model(root_dir + '/results/' + classifier + '/' + archive_name + itr + '/' + dataset_name
                                    + '/best_model.hdf5')

    # calculate accuracy
    y_pred = np.argmax(model.predict(x_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print('overall accuracy:', accuracy)

    # calculate accuracy but exclude each dimension once
    dimension = x_test.shape[2]
    scores = np.zeros(dimension)
    for i in range(dimension):
        x_without_dim_i = np.copy(x_test)
        x_without_dim_i[:, :, i] = 0
        y_pre_without_dim = np.argmax(model.predict(x_without_dim_i), axis=1)
        scores[i] = accuracy_score(y_test, y_pre_without_dim)

    print('occlusion scores:', scores)

    accuracy_lost = (accuracy-scores)*100
    print('accuracy_lost:', accuracy_lost)

    # create plot
    X = np.arange(0, dimension)
    plt.bar(X, accuracy_lost, width=0.5)
    plt.ylabel("accuracy lost [%]")
    plt.xlabel("dimension")
    plt.title("Importance of different dimensions")
    plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + dataset_name + '-occlusion-' + '.png', bbox_inches='tight', dpi=1080)
    plt.close()

    return scores
