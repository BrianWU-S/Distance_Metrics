import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import os
from metric_learn import MMC


def PCA_preprocessing(n_components=50):
    # data import
    X_features = pd.read_csv(r"Dataset\features\ResNet101\AwA2-features.txt", header=None, sep=' ')
    Y_labels = pd.read_csv(r"Dataset\features\ResNet101\AwA2-labels.txt", header=None, sep=' ')
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_features, Y_labels, test_size=0.4,
                                                                        random_state=1234, stratify=Y_labels)
    pca_method = PCA(n_components=50)
    pca_method.fit(X_train)
    pca_reduced_features = pca_method.transform(X_features)
    reduced_x_train, reduced_x_test, reduced_y_train, reduced_y_test = model_selection.train_test_split(
        pca_reduced_features, Y_labels, test_size=0.4, random_state=1234, stratify=Y_labels)
    np.save(r'Dataset\reduced_x_train', reduced_x_train)
    np.save(r'Dataset\reduced_y_train', reduced_y_train)
    np.save(r'Dataset\reduced_x_test', reduced_x_test)
    np.save(r'Dataset\reduced_y_test', reduced_y_test)


def pair_construction():
    pca_reduced_x_train = np.load(r'Dataset\reduced_x_train.npy')
    pca_reduced_x_test = np.load(r'Dataset\reduced_x_test.npy')
    pca_reduced_y_train = np.load(r'Dataset\reduced_y_train.npy')
    pca_reduced_y_test = np.load(r'Dataset\reduced_y_test.npy')
    np.random.seed = 1
    # permutated data
    permutated_pca_reduced_x_train = np.random.permutation(pca_reduced_x_train)
    permutated_pca_reduced_y_train = np.random.permutation(pca_reduced_y_train)
    p_sorted_pca_reduced_y_train = np.sort(np.squeeze(permutated_pca_reduced_y_train))
    p_sort_indices = np.argsort(np.squeeze(permutated_pca_reduced_y_train))
    p_sorted_pca_reduced_x_train = permutated_pca_reduced_x_train[p_sort_indices, :]
    # non-permutated data
    sorted_pca_reduced_y_train = np.sort(np.squeeze(pca_reduced_y_train))
    sort_indices = np.argsort(np.squeeze(pca_reduced_y_train))
    sorted_pca_reduced_x_train = pca_reduced_x_train[sort_indices, :]
    # Dissimilar pairs construction
    a_train, _, a_y_train, _ = model_selection.train_test_split(sorted_pca_reduced_x_train, sorted_pca_reduced_y_train,
                                                                test_size=0.32, random_state=1,
                                                                stratify=sorted_pca_reduced_y_train)
    b_train, _, b_y_train, _ = model_selection.train_test_split(sorted_pca_reduced_x_train, sorted_pca_reduced_y_train,
                                                                test_size=0.32, random_state=2,
                                                                stratify=sorted_pca_reduced_y_train)
    train_pairs_dissimilar = []
    for row in range(a_train.shape[0]):
        tmp_pair = [a_train[row].tolist(), b_train[row].tolist()]
        train_pairs_dissimilar.append(tmp_pair)
    y_pairs_dissimilar = -1 * np.ones(np.shape(train_pairs_dissimilar)[0])
    # Similar pairs construction
    b_train_si, _, b_y_train_si, _ = model_selection.train_test_split(p_sorted_pca_reduced_x_train,
                                                                      p_sorted_pca_reduced_y_train, test_size=0.32,
                                                                      random_state=1,
                                                                      stratify=p_sorted_pca_reduced_y_train)
    train_pairs_similar = []
    for row in range(a_train.shape[0]):
        tmp_pair = [a_train[row].tolist(), b_train_si[row].tolist()]
        train_pairs_similar.append(tmp_pair)
    y_pairs_similar = np.ones(np.shape(train_pairs_similar)[0])
    # Train pairs construction
    train_pairs = np.vstack((train_pairs_dissimilar, train_pairs_similar))
    y_pairs = np.vstack((np.reshape(y_pairs_dissimilar, (-1, 1)), np.reshape(y_pairs_similar, (-1, 1))))
    # save the data as .npy
    np.save(r'Dataset\train_pairs', train_pairs)
    np.save(r'Dataset\y_pairs', y_pairs)
    np.save(r'Dataset\train_pairs_dissimilar', train_pairs_dissimilar)
    np.save(r'Dataset\train_pairs_similar', train_pairs_similar)
    np.save(r'Dataset\y_pairs_dissimilar', y_pairs_dissimilar)
    np.save(r'Dataset\y_pairs_similar', y_pairs_similar)


def MMC_experiment(K_list):
    PCA_reduced_x_train = np.load(r'Dataset\reduced_x_train.npy')
    PCA_reduced_x_test = np.load(r'Dataset\reduced_x_test.npy')
    PCA_reduced_y_train = np.load(r'Dataset\reduced_y_train.npy')
    PCA_reduced_y_test = np.load(r'Dataset\reduced_y_test.npy')
    train_pairs = np.load(r'Dataset\train_pairs.npy')
    y_pairs = np.load(r'Dataset\y_pairs.npy')

    mmc_acc_list = []
    mmc = MMC(random_state=1234, max_iter=100, convergence_threshold=1e-3, verbose=True)
    mmc.fit(pairs=train_pairs, y=np.squeeze(y_pairs))
    mmc_metric = mmc.get_metric()
    for K in K_list:
        knn_mmc = KNeighborsClassifier(n_neighbors=K, metric=mmc_metric)
        knn_mmc.fit(PCA_reduced_x_train, PCA_reduced_y_train)
        Y_pred_mmc = knn_mmc.predict(PCA_reduced_x_test)
        mmc_acc_list.append(metrics.accuracy_score(PCA_reduced_y_test, Y_pred_mmc))
        print("n_neighbours||K: ", K, "Accuracy:", mmc_acc_list[-1])
    mmc_plotting(mmc_acc_list)


def mmc_plotting(acc_list):
    print(acc_list)
    plt.figure(figsize=[20, 10])
    plt.plot(K_list, acc_list)
    plt.scatter(K_list, acc_list)
    plt.xlabel("N_neighbours", fontdict={'size': 18})
    plt.ylabel("Accuracy", fontdict={'size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("MMC", fontsize=18)
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(r'Dataset\reduced_x_train.npy'):
        PCA_preprocessing()
    if not os.path.exists(r'Dataset\train_pairs.npy'):
        pair_construction()

    K_list = [2, 5, 8, 10, 15, 20, 25]
    MMC_experiment(K_list)
    