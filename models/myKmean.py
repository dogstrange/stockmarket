import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """Randomly initialize centroids"""
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[indices]

    def assign_clusters(self, X):
        """Assign each point to the nearest centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        """Update centroids as the mean of points in each cluster"""
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        """Fit the K-means model"""
        self.initialize_centroids(X)

        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()

            self.labels = self.assign_clusters(X)
            self.centroids = self.update_centroids(X, self.labels)

            # Check for convergence
            if np.all(np.linalg.norm(self.centroids - old_centroids, axis=1) < self.tol):
                break

        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X)

    def inertia(self, X):
        """Calculate within-cluster sum of squares (inertia)"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

    def silhouette_score(self, X):
        """Calculate silhouette score"""
        return silhouette_score(X, self.labels)

    def plot_clusters(self, X, title="K-Means Clustering"):
        """Plot the clusters (for 2D data)"""
        if X.shape[1] != 2:
            print("Plotting only works for 2D data")
            return

        plt.figure(figsize=(10, 6))
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       label=f'Cluster {i+1}', alpha=0.6)

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                   marker='x', s=200, c='red', label='Centroids')
        plt.title(title)
        plt.legend()
        plt.show()