from .ass_4 import evaluation_metrics_1, evaluation_metrics_2, preprocess_data, display_scatter_plot

import numpy as np
import matplotlib.pyplot as plt
import sys


# function to plot the selected centroids
def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color='red', label='next centroid')
    plt.title('Select % d centroid' % (centroids.shape[0]))

    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()


# function to compute euclidean distance
def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


# initialization algorithm
def initialize(data, k):
    centroids = []
    centroids.append(data[np.random.randint(
        data.shape[0]), :])
    plot(data, np.array(centroids))

    # compute remaining k - 1 centroids
    for c_id in range(k - 1):
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        plot(data, np.array(centroids))
    return centroids


if __name__ == '__main__':
    # call the initialize function to get the centroids
    X, y = preprocess_data('iris')
    centroids = initialize(X, k=3)
    centroids = np.asarray(centroids)

    # Fit KMeans Algo using initial centroids obtained above
    from sklearn.cluster import KMeans
    kmeans_pp = KMeans(n_clusters=3, init=centroids)
    kmeans_pp.fit(X)

    display_scatter_plot(X, kmeans_pp, True)
    evaluation_metrics_1(X, kmeans_pp)
    evaluation_metrics_2(X, kmeans_pp)



