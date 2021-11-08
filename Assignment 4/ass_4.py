import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans

from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def preprocess_data(key):
    data_dict = {
        'iris': load_iris(),
        'wine': load_wine()
    }

    data = data_dict[key]
    X = scale(data.data)
    y = data.target

    return X, y


def display_scatter_plot(X, model=None, clustered=False):
    if not clustered:
        plt.scatter(X[:,0], X[:,1], label='True Position')
        plt.show()
    else:
        plt.scatter(X[:,0], X[:,1], c=model.labels_, cmap='rainbow')
        plt.show()


def evaluation_metrics_1(X, model):
    print("Evaluation Metrics:")
    print("-----------------------------------")
    print("-----------------------------------")
    print(f"Silhouette Score: {silhouette_score(X, model.labels_)}")
    print("-----------------------------------")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X, model.labels_)}")
    print("-----------------------------------")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X, model.labels_)}")
    print("-----------------------------------")


def evaluation_metrics_2(X, model, kmeans=True):
    from scipy.cluster.vq import vq
    if kmeans:
        codebook = model.cluster_centers_
    else:
        codebook = []
        for i in np.unique(model.labels_):
            cluster_data = X[model.labels_ == i]
            centroid = cluster_data.mean(0)
            codebook.append(centroid)

    partition, euc_distance_to_centroids = vq(X, codebook)

    tss = np.sum((X-X.mean(0))**2)
    sse = np.sum(euc_distance_to_centroids**2)
    ssb = tss - sse
    print("Cohesion Score: ", sse)
    print('-----------------------------------')
    print("Seperation Score", ssb)
    print('-----------------------------------')


def create_dendrogram(X, y):
    from scipy.cluster.hierarchy import dendrogram, linkage
    linked = linkage(X, 'single')
    labelList = range(X.shape[0]+1)

    dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
    plt.savefig("dgram_iris.png")


if __name__ == "__main__":
    print("IRIS DATASET")
    # print("WINE DATASET")

    X, y = preprocess_data('iris')
    # X, y = preprocess_data('wine')
    display_scatter_plot(X)
    n_clusters = 3


    print("KMEANS")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    display_scatter_plot(X, kmeans, True)
    evaluation_metrics_1(X=X, model=kmeans)
    evaluation_metrics_2(X=X, model=kmeans)

    # print("KMEDOIDS")
    # from sklearn_extra.cluster import KMedoids
    # kmeds = KMedoids(n_clusters=n_clusters).fit(X)
    # display_scatter_plot(X, kmeds, True)
    # evaluation_metrics_1(X=X, model=kmeds)
    # evaluation_metrics_2(X=X, model=kmeds)

    print("AGNES")
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(X)
    display_scatter_plot(X, cluster, True)
    evaluation_metrics_1(X=X, model=cluster)
    evaluation_metrics_2(X=X, model=cluster, kmeans=False)

    print("BIRCH")
    from sklearn.cluster import Birch
    birch = Birch(n_clusters=n_clusters).fit(X)
    display_scatter_plot(X, birch, True)
    evaluation_metrics_1(X=X, model=birch)
    evaluation_metrics_2(X=X, model=birch, kmeans=False)

    print("DBSCAN")
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.6, min_samples=8).fit(X)
    display_scatter_plot(X, dbscan, True)
    evaluation_metrics_1(X=X, model=dbscan)
    evaluation_metrics_2(X=X, model=dbscan, kmeans=False)

    print("OPTICS")
    from sklearn.cluster import OPTICS
    optics = OPTICS(min_samples=8, max_eps=0.6).fit(X)
    display_scatter_plot(X, optics, True)
    evaluation_metrics_1(X=X, model=optics)
    evaluation_metrics_2(X=X, model=optics, kmeans=False)
