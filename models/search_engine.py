import numpy as np
from sklearn.cluster import KMeans
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class VectorSearchEngine:
    def __init__(self, data):
        """
        Initializes the search engine with the base dataset.
        data: numpy array of shape (N, D) where N is number of vectors, D is dimensions.
        """
        self.data = data
        self.N, self.D = data.shape
        self.centroids = None
        self.inverted_index = {}
        
        # PCA to project data to 2D
        self.pca = PCA(n_components=2, random_state=42)
        self.data_2d = self.pca.fit_transform(self.data)
            
    def exact_knn(self, query, k=10):
        """
        The Exact Algorithm: Calculates distance to ALL vectors.
        """
        if self.centroids is None:
            raise ValueError("Inverted index not built. Call build_inverted_index() first.")
            
        # Calculate squared Euclidean distances to all vectors
        distances = np.sum((self.data - query) ** 2, axis=1)
        
        # Get the indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]
        
        return nearest_indices, distances[nearest_indices]

    def build_inverted_index(self, num_clusters_p):
        """
        Preprocessing phase: Clusters the data into P groups and builds the index.
        """
        print(f"Building inverted index with P={num_clusters_p} clusters...")
        self.data = self.data_2d
        # Use K-Means to find P centroids and labels of dataset
        kmeans = KMeans(n_clusters=num_clusters_p, random_state=42, n_init=10).fit(self.data)
        self.centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        # Build the inverted index
        self.inverted_index = {i: [] for i in range(num_clusters_p)}
        
        for idx, label in enumerate(labels):
            self.inverted_index[label].append(idx)
            
        for i in range(num_clusters_p):
            cluster_indices = self.inverted_index[i]
            if len(cluster_indices) > 0:
                cluster_vectors = self.data[cluster_indices]
                centroid = self.centroids[i]
                dists_to_centroid = np.sum((cluster_vectors - centroid) ** 2, axis=1)
                sorted_local_idx = np.argsort(dists_to_centroid)
                self.inverted_index[i] = [cluster_indices[idx] for idx in sorted_local_idx]

        print("Inverted index built successfully.")

    def write_index_to_file(self, output_path="output/inverted_index.txt"):
        """
        Outputs the content of the inverted index to a file.
        For each cluster, writes its centroid, size, and the sorted vector indices.
        """
        if self.centroids is None:
            print("Inverted index is empty. Call build_inverted_index() first.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- Inverted Index Content ---\n")
            for cluster_id, indices in self.inverted_index.items():
                centroid = self.centroids[cluster_id]
                centroid_str = ", ".join(f"{val:.4f}" for val in centroid[:5])
                if len(centroid) > 5:
                    centroid_str += ", ..."
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Centroid: [{centroid_str}]\n")
                f.write(f"  Size: {len(indices)} vectors\n")
                f.write(f"  Vector Indices: {indices}\n")
            f.write("------------------------------\n")
        
        print(f"Inverted index successfully written to {output_path}")

    def visualize_clusters(self, queries=None, output_path="output/clusters_visualization.png"):
        """
        Visualizes the data, centroids, and optionally queries in 2D space using PCA. 
        PCA was used to reduce the dimensionality of the data to 2D for visualization purposes.
        Saves the visualization to the specified file.
        """
        if self.centroids is None:
            print("Inverted index is empty. Call build_inverted_index() first.")
            return
        matplotlib.use('Agg')
        # Use pre-computed 2D PCA projection of the dataset and project centroids
        projected_data = self.data_2d
        projected_centroids = self.centroids
        # Transform queries to 2D
        queries = self.pca.transform(queries)

        plt.figure(figsize=(10, 8))
        # Map each data point's index to its cluster label
        labels = np.zeros(len(self.data), dtype=int)
        for cluster_id, indices in self.inverted_index.items():
            for idx in indices:
                labels[idx] = cluster_id

        # Scatter plot for points
        scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap='tab10', alpha=0.6, edgecolors='none', label='Data points')
        
        # Scatter plot for centroids
        plt.scatter(projected_centroids[:, 0], projected_centroids[:, 1], c=range(len(self.centroids)), cmap='tab10', marker='X', s=200, edgecolors='black', linewidths=1.5, label='Centroids')
        
        # Scatter plot for queries if provided
        if queries is not None:
            if queries.shape[1] == 2:
                projected_queries = queries
            else:
                projected_queries = self.pca.transform(queries)
            plt.scatter(projected_queries[:, 0], projected_queries[:, 1], c='red', marker='*', s=150, edgecolors='black', linewidths=1.0, label='Queries')

        plt.title('2D PCA Visualization of Inverted Index Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster ID')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster visualization successfully saved to {output_path}")



    def approximate_knn(self, query, k=10, m_groups=5):
        """
        The Approximate Algorithm: Examines only the M most promising groups.
        """
        if self.centroids is None:
            raise ValueError("Inverted index not built. Call build_inverted_index() first.")
            
        # Find distances from query to all P centroids
        centroid_distances = np.sum((self.centroids - query) ** 2, axis=1)
        
        # Select the M most promising clusters
        best_centroid_indices = np.argsort(centroid_distances)[:m_groups]
        
        # Gather all vector indices from the M clusters
        candidate_indices = []
        for centroid_idx in best_centroid_indices:
            candidate_indices.extend(self.inverted_index[centroid_idx])
            
        # If we have fewer candidates than k, just return what we have
        if len(candidate_indices) == 0:
            return [], []
            
        candidate_indices = np.array(candidate_indices)
        candidate_vectors = self.data[candidate_indices]
        
        # Calculate exact distances ONLY for the candidates
        distances = np.sum((candidate_vectors - query) ** 2, axis=1)
        
        # Get top k
        k = min(k, len(distances))
        top_k_local_indices = np.argsort(distances)[:k]
        
        # Map back to original dataset indices
        final_indices = candidate_indices[top_k_local_indices]
        final_distances = distances[top_k_local_indices]
        
        return final_indices, final_distances

    def write_knn_results(self, queries, output_path="output/knn_results.txt", k=10, m_groups=5):
        """
        Runs both exact_knn and approximate_knn for all queries and writes the results to a file.
        """
        if self.centroids is None:
            raise ValueError("Inverted index not built. Call build_inverted_index() first.")

        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        queries = self.pca.fit_transform(queries)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- KNN Search Results ---\n\n")
            for query_idx, query in enumerate(queries):
                # Run exact KNN
                exact_indices, exact_distances = self.exact_knn(query, k=k)
                
                # Run approximate KNN
                approx_indices, approx_distances = self.approximate_knn(query, k=k, m_groups=m_groups)
                
                # Format distances to 4 decimal places for clean output
                exact_dists_fmt = [round(float(d), 4) for d in exact_distances]
                approx_dists_fmt = [round(float(d), 4) for d in approx_distances]
                
                f.write(f"Query {query_idx}:\n")
                f.write(f"  Exact KNN (k={k}):\n")
                f.write(f"    Indices:   {exact_indices.tolist() if hasattr(exact_indices, 'tolist') else list(exact_indices)}\n")
                f.write(f"    Distances: {exact_dists_fmt}\n")
                f.write(f"  Approximate KNN (k={k}, m_groups={m_groups}):\n")
                f.write(f"    Indices:   {approx_indices.tolist() if hasattr(approx_indices, 'tolist') else list(approx_indices)}\n")
                f.write(f"    Distances: {approx_dists_fmt}\n")
                f.write("\n")
                
        print(f"KNN search results successfully written to {output_path}")