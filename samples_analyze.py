# import modules
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from scipy.interpolate import interp1d


def determine_train_size(X, y):

    # Test_sizes
    test_sizes = [0.15 + 0.025 * i for i in range(24)]  # 24

    # Store accuracies in a dictionary
    accuracies = {}

    # Iterate over different test_sizes
    for k in range(len(test_sizes)):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[k], random_state=42,
                                                            shuffle=True)

        # Random forest classification
        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', verbose=1, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Calculate accuracy
        y_predicted = clf.predict(X_test)
        accuracies[len(X_train)] = accuracy_score(y_test, y_predicted)

    # Print and plot results
    print('accuracies:', accuracies)
    plt.plot(accuracies.keys(), accuracies.values())
    plt.xlabel('number of samples used for training')
    plt.ylabel('accuracy')
    plt.savefig(root_dir + '/plots/' + '-samples_vs_accuracy-' + '.png', dpi=1080)

    print('DONE')


def plot_sample(X, y, sample, channel_name_train):
    # plot a multivariate sample of the EEG data
    fig, axs = plt.subplots(21, 1)
    label = int(y[sample])
    print('label', label)
    i = 0
    for ax in axs.flat:
        if i < 4:
            ax.plot(X[sample, :, i], color='red')
        elif 4 <= i < 6:
            ax.plot(X[sample, :, i], color='b')
        elif 6 <= i < 8:
            ax.plot(X[sample, :, i], color='red')
        elif 8 <= i < 10:
            ax.plot(X[sample, :, i], color='b')
        elif 10 <= i < 12:
            ax.plot(X[sample, :, i], color='red')
        elif 12 <= i < 16:
            ax.plot(X[sample, :, i], color='b')
        elif 16 <= i < 19:
            ax.plot(X[sample, :, i], color='red')
        elif 19 <= i:
            ax.plot(X[sample, :, i], color='black')
        ax.set_ylabel(channel_name_train[i])
        ax.axes.yaxis.set_ticklabels([])
        i = i + 1
    plt.show()


def plot_con(x_train, y_train, root_dir, archive_name, dataset_name, classifier, itr, dimension, idx, idx_filter=0):

    # Import modules
    from tensorflow import keras

    # If uni-variate
    if len(x_train.shape) == 2:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Load model
    model = keras.models.load_model(
        root_dir + '/results/' + classifier + '/' + archive_name + itr + '/' + dataset_name + '/best_model.hdf5')

    # Get filter weights from the first layer
    layer = 1
    filters = model.layers[layer].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[layer].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    colors = [(255 / 255, 160 / 255, 14 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    print('filters shape', filters.shape)

    # Plot convolutions
    convolved_filter_1 = new_feed_forward([x_train])[0]

    clas = int(y_train[idx])-1
    print('class:', y_train[idx])

    # if uni-variate
    if len(x_train.shape) == 2:
        plt.plot(x_train[idx], color=colors[0], label='class' + str(clas+1) + '-raw')
    else:
        plt.plot(x_train[idx, :, dimension], color=colors[0], label='class' + str(clas+1) + '-raw')  # plot sample

    conv = np.convolve(x_train[idx, :, dimension], filters[:, dimension, idx_filter])/(0.01*len(x_train[idx, :, dimension]))
    plt.plot(conv, color=colors_conv[0], label='class' + str(clas+1) + '-conv for this specific channel')
    plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[1], label='class' + str(clas + 1) + '-conv over all channels')

    plt.plot(filters[:, dimension, idx_filter], label='filter')  # plot the filter shape
    plt.legend()

    # Add labels
    plt.xlabel('x')
    plt.ylabel('y')

    # Save result
    plt.savefig(root_dir + '-convolution-' + dataset_name + '-' + classifier + str(idx_filter) + '-' + str(dimension) +
                '-' + str(idx) + '.pdf')
    plt.close()

    return 1


def viz_cam(x_train, y_train, root_dir, archive_name, dataset_name, classifier, itr, sample=280):
    #  the basis for this function was taken from https: // github.com / hfawaz / dl - 4 - tsc.
    #  we adapted the code for our specific scenario and for multivariate input data
    # import necessary modules
    from tensorflow import keras

    max_length = 2000

    # uni-variate
    if len(x_train.shape) == 2:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # load model (from deep learning file)
    model = keras.models.load_model(
        root_dir + '/results/' + classifier + '/' + archive_name + itr + '/' + dataset_name
        + '/best_model.hdf5')

    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs

    # output is both the original and the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    # sample to investigate
    # sample = 280  # 6030
    print('class:', y_train[sample])

    # channel names
    channel_name_train = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',
                          'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                          'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ',
                          'PZ', 'A1', 'A2']

    # create multivariate figure
    fig, axs = plt.subplots(21, 1, sharex=True)
    dim = 0
    for ax in axs.flat:
        ts = x_train[sample, :, :]
        ts = ts[None, :, :]
        [conv_out, predicted] = new_feed_forward([ts])
        pred_label = np.argmax(predicted, axis=1)
        orig_label = int(y_train[sample])

        if pred_label == orig_label:
            cas = np.zeros(dtype=np.float64, shape=(conv_out.shape[1]))
            for k, w in enumerate(w_k_c[:, orig_label]):
                cas += w * conv_out[0, :, k]
            minimum = np.min(cas)

            cas = cas - minimum

            cas = cas / max(cas)
            cas = cas * 100

            x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)

            # linear interpolation to smooth
            f = interp1d(range(ts.shape[1]), ts[0, :, dim])
            y = f(x)
            # if (y < -2.2).any():
            #     continue
            f = interp1d(range(ts.shape[1]), cas)
            cas = f(x).astype(int)
            im = ax.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.5)
            if dataset_name == 'Gun_Point':
                if orig_label == 1:
                    ax.yticks([-1.0, 0.0, 1.0, 2.0])
                else:
                    ax.yticks([-2, -1.0, 0.0, 1.0, 2.0])
            ax.set_ylabel(channel_name_train[dim])
            dim = dim + 1

    # Save result
    plt.colorbar(im, ax=axs.ravel().tolist())
    plt.show()


# main
if __name__ == '__main__':

    # Determine test size and plotting for multivariate EEG data

    archive_name = 'EyeMovements'
    dataset_name = 'EyeMovements'
    classifier_name = 'resnet'
    itr = ''

    sample = 9000  # 234 and 9000 chosen as example

    # Change for your directory
    root_dir = '/Users/leluwy/Desktop/ETH/Bachelorprojekt/Multivariate_arff'

    # Multi-variate eye-movement dataset
    file_name = root_dir + '/' + dataset_name + '/'
    X_train = np.load(file_name + 'X_train.npy')
    y = np.load(file_name + 'y_train.npy')

    # Standardization
    std_ = X_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (X_train - X_train.mean(axis=1, keepdims=True)) / std_

    # number of samples
    print('number of samples:', len(x_train))

    X = x_train

    # channel names for plotting
    channel_name_train = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',
                          'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                          'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ',
                          'PZ', 'A1', 'A2']
    # Plot sample
    plot_sample(X, y, sample, channel_name_train)

    # Deep Learning Class Activation Map with all channels plotted
    # viz_cam(x_train, y, root_dir, archive_name, dataset_name, classifier_name, itr, sample=sample)

    # Plot convolutions
    plot_con(x_train, y, root_dir, archive_name, dataset_name, classifier='cnn', itr=itr, dimension=0, idx_filter=4,
             idx=sample)

    # Reshape X into a 2D array
    X = X.reshape(X.shape[0], -1)

    # determine_train_size(X, y)  # determine train size
