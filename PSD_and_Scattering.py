# import necessary modules
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# change this directory for your machine
root_dir = '/Users/leluwy/Desktop/ETH/RandomForest/'


def principal_component_analysis(features_scaled, max_components):
    # apply PCA to the scaled features
    pca = PCA(n_components=max_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print('new feature shape:', features_pca.shape)
    return features_pca


def delete_channel(X, channels):
    # delete channels in the channels array
    x_deleted = X.copy()
    for i in range(len(channels)):
        x_deleted = np.concatenate((x_deleted[:, :, 0:channels[i]], x_deleted[:, :, channels[i] + 1:]), axis=2)
        for k in range(len(channels)):
            if channels[k] > channels[i]:
                channels[k] = channels[k] - 1
    return x_deleted


def print_relevance(levels):

    # determine the channel and power band for each entry
    channel = np.floor(levels.index % 21)
    power_band = np.floor(levels.index) / 21

    # print relevance of specific frequency bands
    print('delta', np.sum(levels[power_band == 0]))
    print('theta', np.sum(levels[power_band == 1]))
    print('alpha', np.sum(levels[power_band == 2]))
    print('beta', np.sum(levels[power_band == 3]))
    print('gamma', np.sum(levels[power_band == 4]))

    # Print relevance of specific channels
    for i in range(21):
        print('channel' + str(i), np.sum(levels[channel == i]))


def mutual_info_selection(power_feature, y, max):

    # Convert to pandas DataFrame
    power_feature_dataframe = pd.DataFrame(power_feature)

    # Information gain
    mutual_info = mutual_info_classif(power_feature_dataframe, y, random_state=42)
    mutual_info = pd.Series(mutual_info)
    mutual_info_sorted = mutual_info.sort_values(ascending=False)
    index = mutual_info_sorted[0:max].index

    # Get name
    channel = (np.floor(mutual_info_sorted.index % 21))
    sub_band = np.floor(mutual_info_sorted.index / 21)

    x_label = ['c' + str(int(channel[l])) + 'b' + str(int(sub_band[l])) for l in range(max)]
    plt.bar(x_label, mutual_info_sorted[0:max])
    plt.ylabel('FEATURE IMPORTANCE')
    plt.xlabel('FEATURES')
    plt.show()
    return power_feature[:, index]


def mutual_info_analysis(power_feature, y, band):

    # Convert to pandas DataFrame
    power_feature_dataframe = pd.DataFrame(power_feature)

    # Information gain
    mutual_info = mutual_info_classif(power_feature_dataframe, y, random_state=42)
    mutual_info = pd.Series(mutual_info)

    print_relevance(mutual_info)  # output the relevance for specific channels and bands

    channel_name_train = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',
                          'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                          'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ',
                          'PZ', 'A1', 'A2']

    power_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Plot a specific frequency band
    incr = band * 21

    mutual_info_new = mutual_info[(0 + incr):(21 + incr)]
    mutual_info_new.index = channel_name_train
    mutual_info_sorted = mutual_info_new.sort_values(ascending=False)

    plt.bar(mutual_info_sorted.index, mutual_info_sorted)
    plt.ylim([0, 0.07])
    plt.ylabel('mutual_info')
    plt.xlabel(power_bands[band])
    plt.show()


def random_forest(X_train, X_test, y_train, y_test, repetitions, feature_importance=True):

    accuracies = np.zeros(repetitions)
    precision = np.zeros(repetitions)
    recall = np.zeros(repetitions)
    f1 = np.zeros(repetitions)

    # Repeat experiment to account for the randomness involved (or use random state)
    for k in range(repetitions):

        # Random forest
        clf = RandomForestClassifier(n_estimators=1000, verbose=1, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)

        # Predict test set
        y_predicted = clf.predict(X_test)

        # Metrics
        accuracies[k] = accuracy_score(y_test, y_predicted)
        precision[k] = precision_score(y_test, y_predicted)
        recall[k] = recall_score(y_test, y_predicted)
        f1[k] = f1_score(y_test, y_predicted)

        # Feature importance plot
        if feature_importance:
            feature_importances = clf.feature_importances_  # extract the RF feature importance
            levels = pd.Series(feature_importances)
            levels = levels.sort_values(ascending=False)

            print_relevance(levels)
            # Plot feature importance
            plt.plot(feature_importances)
            plt.show()

    # Results
    print('accuracies:', accuracies)
    mean_accuracy = np.mean(accuracies)
    variance = np.var(accuracies)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)
    confusion = confusion_matrix(y_test, y_predicted)

    return mean_accuracy, variance, mean_precision, mean_recall, mean_f1, confusion


# main
if __name__ == '__main__':

    # Settings
    repetitions = 1
    plot_feature_relevance = False

    # Import data
    y = np.load(root_dir + 'data/' + 'y_train.npy')
    power_feature = np.load(root_dir + 'data/' + 'power_feature.npy')

    # Read scattering features
    scattering_feature1 = np.load(root_dir + 'scattering_features/' + 'scattering_feature_bior1.3_and_rbio3.1.npy')
    scattering_feature2 = np.load(root_dir + 'scattering_features/' + 'scattering_feature_rbio3.1_and_bior1.3.npy')
    scattering_feature3 = np.load(root_dir + 'scattering_features/' + 'scattering_feature_rbio3.1_and_rbio3.1.npy')

    # Delete specific frequency bands
    delete_bands = []
    power_feature = np.delete(power_feature, delete_bands, axis=1)

    # Delete specific channels
    delete_channels = []
    power_feature = delete_channel(power_feature, delete_channels)

    # Reshape power_feature into a 2D array
    power_feature = power_feature.reshape(power_feature.shape[0], -1)

    features = np.concatenate([scattering_feature3], axis=1)  # select features to be analyzed

    # Standardize
    sc = StandardScaler()
    sc.fit_transform(features)

    # mutual information analysis
    # mutual_info_analysis(features, y, 0)
    # features = mutual_info_selection(features, y, 30)

    print('features shape', features.shape)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.4, random_state=42,
                                                        shuffle=True)

    # Classification
    metrics = random_forest(X_train, X_test, y_train, y_test, repetitions, plot_feature_relevance)

    confusion = metrics[5]  # confusion matrix

    # Print results
    print('accuracy:', metrics[0])
    print('variance:', metrics[1])
    print('precision:', metrics[2])
    print('recall:', metrics[3])
    print('f1:', metrics[4])
    print('confusion matrix:', confusion)

    confusion = confusion / len(X_test)

    # Plot confusion matrix
    sns.set(font_scale=2.5)
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt='.2%', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()
