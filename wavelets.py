# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# Change this directory for your machine
root_dir = '/Users/leluwy/Desktop/ETH/RandomForest/'


def mutual_info_selection(features, y, max):

    print('mutual info analysis')

    # Convert to pandas DataFrame
    feature_dataframe = pd.DataFrame(features)

    # Information gain
    mutual_info = mutual_info_classif(feature_dataframe, y, random_state=42)
    feature_important(mutual_info, 0)

    mutual_info = pd.Series(mutual_info)
    # mutual_info = pd.read_csv(root_dir + 'mutual_info.csv')

    # Sort the MI-score in descending order
    mutual_info_sorted = mutual_info.sort_values(ascending=False)
    index = mutual_info_sorted[0:max].index

    # Select only the most relevant features
    new_features = features[:, index]

    print('new feature shape:', new_features.shape)

    # Save results
    mutual_info.to_csv(root_dir + 'mutual_info.csv', index=False)

    return new_features


def principal_component_analysis(features_scaled, max_components):
    # Apply PCA to the scaled features
    pca = PCA(n_components=max_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print('new feature shape:', features_pca.shape)
    return features_pca


def print_relevance(statistical_features, sub_band, channel, levels):

    # Print the relevance for specific statistical features
    print('min_wavelet_coeff:', np.sum(levels[statistical_features == 0]))
    print('max_wavelet_coeff:', np.sum(levels[statistical_features == 1]))
    print('mean_wavelet_coeff:', np.sum(levels[statistical_features == 2]))
    print('median_wavelet_coeff:', np.sum(levels[statistical_features == 3]))
    print('std_deviation_wavelet_coeff:', np.sum(levels[statistical_features == 4]))
    print('skewness_wavelet_coeff:', np.sum(levels[statistical_features == 5]))
    print('energy_wavelet_coeff:', np.sum(levels[statistical_features == 6]))
    print('relative_energy_wavelet_coeff:', np.sum(levels[statistical_features == 7]))
    print('entropy_wavelet_coeff:', np.sum(levels[statistical_features == 8]))

    # Print the relevance for specific sub-bands
    for i in range(16):
        print('sub_band' + str(i), np.sum(levels[sub_band == i]))

    # print the relevance for specific channels
    for i in range(21):
        print('channel' + str(i), np.sum(levels[channel == i]))


def correlation_analysis(features_scaled, y, number_of_features_used=50):

    # concat features and labels
    features = pd.concat([features_scaled, y], axis=1)

    # calculate correlation matrix
    corr_matrix = features.corr().abs()

    # sort features by correlation to the label
    sorted_features = corr_matrix['label'].sort_values(ascending=False)

    print(sorted_features.head())

    # select only the top features
    top_features = sorted_features.index[1:number_of_features_used+1]
    selected_features = features[top_features]

    return selected_features


def delete_channel(X, channels):
    # delete channels in the channels array
    x_deleted = X.copy()
    for i in range(len(channels)):
        x_deleted = np.concatenate((x_deleted[:, :, 0:channels[i]], x_deleted[:, :, channels[i] + 1:]), axis=2)
        for k in range(len(channels)):
            if channels[k] > channels[i]:
                channels[k] = channels[k] - 1
    return x_deleted


def delete_statistical_features(feature, to_be_deleted):

    # Input: to_be_deleted is an array which contains the indexes of the statistical features that will be deleted

    # delete specific statistical features
    mask = pd.Series(True, index=feature.columns)

    for i in range(len(to_be_deleted)):
        mask[feature.columns[to_be_deleted[i]::9]] = False

    # apply mask to the feature
    feature = feature.loc[:, mask]

    return feature


def feature_important(important, iter, max_features=100):

    # sort
    levels = pd.Series(important)

    levels = levels.sort_values(ascending=False)
    levels = levels[:max_features]

    # get name
    channel = np.floor(levels.index / (16*9))
    sub_band = np.round(levels.index / 9) % 16
    statistical_features = levels.index % 9

    print_relevance(statistical_features, sub_band, channel, levels)  # output relevances

    # plot
    x_label = ['c:' + str(channel[l]) + 'b:' + str(sub_band[l]) + 's:' + str(statistical_features[l]) for l in range(max_features)]
    plt.figure(figsize=(40, 20))
    plt.bar(x_label, height=levels)
    plt.title('Random Forest Feature Importance ' + str(iter))
    plt.ylabel('FEATURE IMPORTANCE')
    plt.xlabel('FEATURES')
    plt.show()
    plt.savefig(root_dir + 'plots/' + '-plot-' + str(iter) + '.png')


def random_forest_analyze(X_train, X_test, y_train, y_test, repetitions, iter, plot_feature_importance=False):

    accuracies = np.zeros(repetitions)
    precision = np.zeros(repetitions)
    recall = np.zeros(repetitions)
    f1 = np.zeros(repetitions)

    # repeat experiment to account for the randomness involved (or use random state)
    for k in range(repetitions):

        # random forest
        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', verbose=1, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)

        # predict test set
        y_predicted = clf.predict(X_test)

        # metrics
        accuracies[k] = accuracy_score(y_test, y_predicted)
        precision[k] = precision_score(y_test, y_predicted)
        recall[k] = recall_score(y_test, y_predicted)
        f1[k] = f1_score(y_test, y_predicted)

    # feature importance plot
    if plot_feature_importance:
        feature_importances = clf.feature_importances_  # RF- feature importance
        feature_important(feature_importances, iter)

    # results
    mean_accuracy = np.mean(accuracies)
    variance = np.var(accuracies)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)
    confusion = confusion_matrix(y_test, y_predicted)

    return mean_accuracy, variance, mean_precision, mean_recall, mean_f1, confusion


def feature_selection(power_feature, y, number_of_features_used=30):
    # reshape
    power_feature = power_feature.reshape(power_feature.shape[0], -1)
    power_feature_dataframe = pd.DataFrame(power_feature)

    # information gain
    mutual_info = mutual_info_classif(power_feature_dataframe, y)
    mutual_info = pd.Series(mutual_info)

    channel_name_train = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                          'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                          'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF',
                          'EEG PZ-REF', 'EEG A1-REF', 'EEG A2-REF']

    power_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # plot a specific band
    plot = False
    band = 0

    if plot:
        incr = band * 21
        mutual_info_new = mutual_info[0 + incr:21 + incr]
        mutual_info_new.index = channel_name_train
        mutual_info_sorted = mutual_info_new.sort_values(ascending=False)
        plt.bar(mutual_info_sorted.index, mutual_info_sorted)
        plt.title('feature_selection_plot')
        plt.ylabel('mutual_info')
        plt.xlabel(power_bands[band])
        plt.show()

    # use only the most relevant features
    power_bands_relevant = mutual_info.sort_values(ascending=False)[0:number_of_features_used]
    power_feature = power_feature[:, power_bands_relevant.index]

    return power_feature


def do_classification(repetitions, features, iter, correlation=False, do_pca=False, mutual_info=False, max_features=100):

    # time
    start = time.time()

    # handling NaN values
    features = features.fillna(-1e9)

    # performing standardization
    sc = StandardScaler()
    features_scaled = sc.fit_transform(features)

    # correlation analysis
    if correlation:
        features_scaled = correlation_analysis(pd.DataFrame(features_scaled), pd.DataFrame(y), max_features)

    # Principal Component Analysis
    if do_pca:
        features_scaled = principal_component_analysis(features_scaled, max_features)

    # Mutual Info Analysis
    if mutual_info:
        features_scaled = mutual_info_selection(features_scaled, y, max_features)

    # train test split
    features = np.array(features_scaled)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.4, random_state=42)

    # classification
    results, variance, precision, recall, f1, confusion = random_forest_analyze(X_train, X_test, y_train, y_test, repetitions,
                                                                         iter)
    # save results
    print('confusion matrix:', confusion)

    confusion = confusion / confusion.sum(axis=1).sum(axis=0)
    # Plot confusion matrix
    sns.set(font_scale=2.5)
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt='.2%', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    results = {'repetitions': [repetitions], 'average accuracy': [results], 'variance': [variance],
                   'precision(PPV)': [precision], 'recall(TPR)': [recall], 'f1': [f1], 'time': [time.time() - start],
                   'TP': [confusion[0][0]], 'FN': [confusion[0][1]], 'FP': [confusion[1][0]], 'TN': [confusion[1][1]]}

    pd_results = pd.DataFrame(results)
    pd_results.to_csv(root_dir + 'pd_results.csv', index=False)
    print(pd_results)

    print('DONE')
    return pd_results


# main
if __name__ == '__main__':

    print('classifier: RandomForest')

    # settings
    classification = True
    do_feature_selection = False
    repetitions = 1

    max_features = 1000  # select the maximum number of features for feature selection

    # features to read
    sym = False
    r_bio = True
    other = False
    db = False
    coif = False
    bior = False

    # import data
    y = np.load(root_dir + 'data/' + 'y_train.npy')
    power_feature = np.load(root_dir + 'data/' + 'power_feature.npy')
    power_feature = power_feature.reshape(power_feature.shape[0], -1)
    power_feature = pd.DataFrame(power_feature)

    # read wavelet features

    # read sym
    if sym:
        sym_dfs = []
        for i in range(2, 21):
            sym_df = pd.DataFrame(np.load(root_dir + f'wavelet_features/sym/wavelet_feature_sym{i}.npy'))
            sym_dfs.append(sym_df)

    # read r-bio
    if r_bio:
        rbio_files = ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6',
                     'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9',
                     'rbio4.4', 'rbio5.5', 'rbio6.8']

        rbio_dfs = []
        for file in rbio_files:
            rbio_df = pd.DataFrame(np.load(f"{root_dir}wavelet_features/rbio/wavelet_feature_{file}.npy"))
            rbio_dfs.append(rbio_df)

    # read other
    if other:
        dmey = pd.DataFrame(np.load(root_dir + 'wavelet_features/other/' + 'wavelet_feature_dmey.npy'))
        haar = pd.DataFrame(np.load(root_dir + 'wavelet_features/other/' + 'wavelet_feature_haar.npy'))

    # read db
    if db:
        db_dfs = []
        for i in range(1, 39):
            db_df = pd.DataFrame(np.load(root_dir + f"wavelet_features/db/wavelet_feature_db{i}.npy"))
            db_dfs.append(db_df)

    # read coif
    if coif:
        coifs_dfs = []
        for i in range(1, 18):
            coif_df = pd.DataFrame(np.load(root_dir + f"wavelet_features/coif/wavelet_feature_coif{i}.npy"))
            coifs_dfs.append(coif_df)

    # read bior
    if bior:
        bior_files = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3',
                      'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
        bior_dfs = []
        for file in bior_files:
            bior_df = pd.DataFrame(np.load(f"{root_dir}wavelet_features/bior/wavelet_feature_{file}.npy"))
            bior_dfs.append(bior_df)

    # select feature combinations
    features_con = [rbio_dfs[7]]  # rbio_dfs[7]

    # iterate through all feature combinations
    if classification:
        result_matrix = pd.DataFrame()
        for i in range(len(features_con)):
            features = features_con[i]
            print('feature:', i)
            print('feature shape', features.shape)
            result = do_classification(repetitions=repetitions, features=features, iter=i, max_features=max_features,
                                       correlation=False, do_pca=False, mutual_info=False, )
            result_matrix = pd.concat([result_matrix, result], axis=0)
            # Save results
            result_matrix.to_csv(root_dir + 'pd_results_matrix.csv', index=False)
