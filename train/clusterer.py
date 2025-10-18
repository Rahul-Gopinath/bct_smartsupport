import json
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

class Clusterer:
    def __init__(self):
        self.clusterer = hdbscan.HDBSCAN (
                         min_cluster_size=5,       # tweak based on expected cluster density
                         metric='precomputed',     # since we provide a distance matrix
                         cluster_selection_epsilon=0.01, # small epsilon for fine-grained clusters
                         cluster_selection_method='eom'
                        )
    def cluster(self, embeddings):
        """
        Accepts a NumPy array of embeddings.
        Returns cluster labels for each embedding.
        """
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        distance_matrix = cosine_distances(normed).astype(np.float64)
        cluster_labels = self.clusterer.fit_predict(distance_matrix)
        return cluster_labels
        
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', help = 'model_file', type = str)#, default = './Log2VecOutput/log2vec_words.model')
    parser.add_argument('--output', help = 'clustered_segments', type = str, default='models/clustered_segments.json')
    args = parser.parse_args()

    clustered_segments = args.output
    
    embeddings = np.load('models/template_embeddings.npy')
    clusterer = Clusterer()
    embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Ensure 2D shape
    print("Reshaped Embedding shape:", embeddings.shape)
    labels = clusterer.cluster(embeddings)
    print("Cluster labels:", labels)
    print("Total data points:", len(labels))
    print("Noise points (-1 labels):", np.sum(labels == -1))
    print("Clustered points:", np.sum(labels != -1))
    print("Number of clusters found:", len(set(labels)) - (1 if -1 in labels else 0))

    # Save clustered segments to a JSON file
    clustered_data = [{"embedding_index": i, "cluster_label": int(label)} for i, label in enumerate(labels)]
    with open(clustered_segments, "w") as f:
        json.dump(clustered_data, f, indent=2)