from .ass_4 import preprocess_data

import numpy as np
import matplotlib.pyplot as plt


def plot(data, centroids, next_centroid):
    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                color='black', label='previously selected centroids')
    plt.scatter(next_centroid[0], next_centroid[1],
                color='red', label='next centroid')
    plt.title('Iteration: % d' % (centroids.shape[0]))

    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 5)
    plt.show()


def bisecting_kmeans(X, n_clusters):
    K = n_clusters
    current_clusters = 1
    centroids = []
    clusters = []
    X_ = X

    while current_clusters != K:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2).fit(X_)
        current_clusters += 1
        cluster_centers = kmeans.cluster_centers_

        # identify next cluster for kmeans
        sse = [0] * 2
        for point, label in zip(X_, kmeans.labels_):
            sse[label] += np.square(point - cluster_centers[label]).sum()
        chosen_cluster = np.argmax(sse, axis=0)

        # save the other cluster
        centroids.append(cluster_centers[1 - chosen_cluster])
        clusters.append(X_[kmeans.labels_ == 1 - chosen_cluster])

        # reassign chosen cluster for next iteration
        plot(X_, np.array(centroids), cluster_centers[chosen_cluster])
        chosen_cluster_data = X_[kmeans.labels_ == chosen_cluster]
        X_ = chosen_cluster_data

    centroids.append(X_.mean(0))
    clusters.append(X_)

    return centroids, clusters


if __name__ == '__main__':
    # call the initialize function to get the centroids
    X, y = preprocess_data('iris')
    centroids, clusters = bisecting_kmeans(X, n_clusters=3)

    # print(centroids)
    # print(clusters)

    # make labels array
    labels = np.zeros(X.shape[0])
    labels[clusters[0].shape[0]:clusters[0].shape[0] + clusters[1].shape[0]] = 1
    labels[clusters[0].shape[0] + clusters[1].shape[0]:] = 2

    # get data back from the clusters
    X_ = np.array([item for sublist in clusters for item in sublist])

    # display scatter plot
    plt.scatter(X_[:, 0], X_[:, 1], c=labels, cmap='rainbow')
    plt.show()

    # Evaluation Metrics 1
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    print("Evaluation Metrics:")
    print("-----------------------------------")
    print("-----------------------------------")
    print(f"Silhouette Score: {silhouette_score(X_, labels)}")
    print("-----------------------------------")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_, labels)}")
    print("-----------------------------------")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X_, labels)}")
    print("-----------------------------------")

    # Evaluation Metrics 2
    from scipy.cluster.vq import vq
    codebook = centroids
    partition, euc_distance_to_centroids = vq(X_, np.array(codebook))
    tss = np.sum((X_ - X_.mean(0)) ** 2)
    sse = np.sum(euc_distance_to_centroids ** 2)
    ssb = tss - sse
    print("Cohesion Score: ", sse)
    print('-----------------------------------')
    print("Seperation Score", ssb)
    print('-----------------------------------')



