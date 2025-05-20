import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import DictionaryLearning, SparseCoder, PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def calculate_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm."""
    w = np.zeros((y_pred.max() + 1, y_true.max() + 1))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return w[ind].sum() / y_pred.size


def create_matrix(X, f, gamma, max_value=1e10):
    """Create Similarity matrix based on a distance matrix and pheromone matrix."""

    # Calculate the distance matrix (Euclidean distances)
    distance_matrix = cdist(X, X, 'euclidean')

    # Apply the transformation
    Transformed_matrix = np.exp(-distance_matrix / np.abs(f))

    # Handle potential overflow by clipping large values
    Transformed_matrix = np.clip(Transformed_matrix, None, max_value)

    # Handle NaNs or Infinities by replacing them with a large number or the mean value
    if np.any(np.isnan(Transformed_matrix)) or np.any(np.isinf(Transformed_matrix)):
        mean_value = np.nanmean(Transformed_matrix[np.isfinite(Transformed_matrix)])
        Transformed_matrix = np.nan_to_num(Transformed_matrix, nan=mean_value, posinf=max_value, neginf=max_value)

    # Ensure symmetry by averaging the matrix with its transpose
    return (Transformed_matrix + Transformed_matrix.T) / 2

def fitness_sparse(X, cluster_labels, dictionary):
    """Calculate fitness based on sparse representation."""
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars')
    sparse_codes = coder.transform(X)

    # Compute within-cluster variance
    within_cluster_variance = sum(
        np.sum((sparse_codes[cluster_labels == cluster] - sparse_codes[cluster_labels == cluster].mean(axis=0)) ** 2)
        for cluster in np.unique(cluster_labels)
    )
    return -within_cluster_variance  # Negate to maximize fitness

def evaluate_clustering(X, labels):
    """Evaluate clustering results using various metrics."""
    return {
        'Silhouette': round(silhouette_score(X, labels),4),
        'DBI': round(davies_bouldin_score(X, labels),4),
        'CH Index': round(calinski_harabasz_score(X, labels),4),
    }
def main():
    # 读取数据
    df = pd.read_excel('molecule.xlsx')
    cid = df['CID']
    X = df.iloc[:, 1:]
    print(X)
    print(X.shape)
    print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    gamma = 1.0 / X_scaled.shape[1]
    start_time = time.time()
    # Initial Transformed_matrix
    f0 = 1
    Similarity_matrix_initial = create_matrix(X_scaled, f0, gamma)
    print(np.isinf(Similarity_matrix_initial))
    # Dictionary learning for sparse coding
    dictionary_learner = DictionaryLearning(n_components=X_scaled.shape[1], random_state=SEED)
    dictionary_matrix = dictionary_learner.fit(X_scaled).components_
    clustering_initial = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=SEED)
    cluster_labels_initial = clustering_initial.fit_predict(Similarity_matrix_initial)
    f = fitness_sparse(X_scaled, cluster_labels_initial, dictionary_matrix)
    Similarity_matrix = create_matrix(X_scaled, f, gamma)
    # Clustering after optimization
    clustering_after = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=SEED)
    labels_after = clustering_after.fit_predict(Similarity_matrix)
    runtime_after = time.time() - start_time
    print("\nAfter Optimization:")
    print("Cluster distribution:", Counter(labels_after))
    # Evaluate after optimization
    metrics_after = evaluate_clustering(Similarity_matrix, labels_after)
    print(f"Metrics: {metrics_after}, Runtime: {runtime_after:.4f} seconds")
    df['Cluster'] = labels_after
    output_file = 'outputClusteringLabel.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Clustering results saved to {output_file}")
    # PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(X_scaled)

    # cluster_colors = {0: '#7F9FAF', 1: '#E2A2AC', 2 : "#284852"}
    cluster_colors = {0: '#F2B2AC', 1: '#8FC4D9', 2: "#CEB4A5"}
    highlight_color = 'red'  # Highlight CID 7005
    highlight_index = df[df['CID'] == 7005].index
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(12, 8))
    for i in range(len(X_scaled)):
        if i not in highlight_index:
            color = cluster_colors[labels_after[i]]
            plt.scatter(data_pca[i, 0], data_pca[i, 1], color=color, s=50, alpha=0.7)
    for i in highlight_index:
        plt.scatter(data_pca[i, 0], data_pca[i, 1], color=highlight_color, s=150, alpha=1.0, marker='*',
                    edgecolors=highlight_color, linewidth=1.5, label='Target Molecule')
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.savefig('ASC_MS.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()

