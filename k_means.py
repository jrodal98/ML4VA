import numpy as np


def dist(v1, v2):
    return (((v1 - v2)**2).sum(axis=2))**.5


class K_means:
    def __init__(self, values):
        self.values = values
        self.k = 0
        self.closest = None
        self.centroids = None

    def cluster(self, k):
        self.k = k
        centroids = np.random.permutation(self.values)[:k]  # Initialize centroid
        closest = np.empty(self.values.shape[0])  # Initialize the closest array
        previous_closest = None  # keep track of the previous closest array
        while not all(previous_closest == closest):  # while classes are still changing
            previous_closest = closest
            # populate an array with the closest centroid for each row in values
            closest = np.argmin(dist(self.values, centroids[:, np.newaxis]), axis=0)
            # new centroids are the mean point of the values assigned to that centroid
            centroids = np.array([self.values[closest == clust].mean(axis=0)
                                  for clust in range(centroids.shape[0])])
        self.centroids = centroids
        self.closest = closest
        return centroids

    def get_cluster_sds(self):
        return np.array([self.values[self.closest == clust].std(axis=0)
                         for clust in range(self.centroids.shape[0])])

    def squared_err(self):
        return np.array([(self.values[self.closest == clust]**2).sum(axis=0) - self.values[self.closest == clust].shape[0] * self.centroids[clust]**2 for clust in range(self.centroids.shape[0])])

    def get_clusters(self):
        return [self.values[self.closest == clust] for clust in range(self.centroids.shape[0])]
