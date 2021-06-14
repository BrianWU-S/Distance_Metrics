import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import os


def PCA_preprocessing(n_components=50):
    # data import
    X_features = pd.read_csv(r"Dataset\features\ResNet101\AwA2-features.txt", header=None, sep=' ')
    Y_labels = pd.read_csv(r"Dataset\features\ResNet101\AwA2-labels.txt", header=None, sep=' ')
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_features, Y_labels, test_size=0.4,
                                                                        random_state=1234, stratify=Y_labels)
    pca_method = PCA(n_components=50)
    pca_method.fit(X_train)
    PCA_reduced_features = pca_method.transform(X_features)
    reduced_x_train, reduced_x_test, reduced_y_train, reduced_y_test = model_selection.train_test_split(
        PCA_reduced_features, Y_labels, test_size=0.4, random_state=1234, stratify=Y_labels)
    np.save(r'Dataset\reduced_x_train', reduced_x_train)
    np.save(r'Dataset\reduced_y_train', reduced_y_train)
    np.save(r'Dataset\reduced_x_test', reduced_x_test)
    np.save(r'Dataset\reduced_y_test', reduced_y_test)


def custom_dist(x, y):
    cosine_similarity = np.dot(x, y) / (np.multiply(np.sqrt(np.sum(np.square(x))), np.sqrt(np.sum(np.square(y)))))
    return (1 - cosine_similarity) * 0.5


def similarity_experiment(K_list):
    PCA_reduced_x_train = np.load(r'Dataset\reduced_x_train.npy')
    PCA_reduced_x_test = np.load(r'Dataset\reduced_x_test.npy')
    PCA_reduced_y_train = np.load(r'Dataset\reduced_y_train.npy')
    PCA_reduced_y_test = np.load(r'Dataset\reduced_y_test.npy')
    cs_acc_list = []
    for K in K_list:
        knn_cs = KNeighborsClassifier(n_neighbors=K, metric=custom_dist, n_jobs=8)
        knn_cs.fit(PCA_reduced_x_train, PCA_reduced_y_train)
        Y_pred_cs = knn_cs.predict(PCA_reduced_x_test)
        cs_acc_list.append(metrics.accuracy_score(PCA_reduced_y_test, Y_pred_cs))
        print("n_neighbours||K: ", K, "Accuracy:", cs_acc_list[-1])
    cs_plotting(cs_acc_list)


def cs_plotting(cs_acc_list):
    print(cs_acc_list)
    plt.figure(figsize=[20, 10])
    plt.plot(K_list, cs_acc_list)
    plt.scatter(K_list, cs_acc_list)
    plt.xlabel("N_neighbours", fontdict={'size': 18})
    plt.ylabel("Accuracy", fontdict={'size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Cosine Similarity", fontsize=18)
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(r'Dataset\reduced_x_train.npy'):
        PCA_preprocessing()
    K_list = [2, 5, 8, 10, 15, 20, 25]
    similarity_experiment(K_list)

