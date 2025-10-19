import numpy as np
import json
import faiss

class Indexer:
    def __init__(self):
        pass

    def normalize_vectors(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-8)

    def compute_centroids(self, embeddings, labels):
        centroids = {}
        for label in np.unique(labels):
            if label == -1: continue
            members = embeddings[labels == label]
            centroid = np.mean(members, axis=0)
            centroids[label] = centroid / np.linalg.norm(centroid)
        return centroids

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.normalize_vectors(embeddings))
        return index

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', help = 'model_file', type = str)#, default = './Log2VecOutput/log2vec_words.model')
    # parser.add_argument('--output', help = 'clustered_segments', type = str, default='clustered_segments.json')
    # args = parser.parse_args()

    print("Loading embeddings and labels...")
    embeddings = np.load('models/template_embeddings.npy')
    with open('models/clustered_segments.json', 'r') as f:
        clustered_data = json.load(f)
        labels = np.array([item['cluster_label'] for item in clustered_data])

    indexer = Indexer()

    centroids = indexer.compute_centroids(embeddings, labels)
    print(f"Computed {len(centroids)} centroids.")
    for label, centroid in centroids.items():
        print(f"Centroid for cluster {label}: {centroid[:5]}...")

    np.save('models/cluster_centroids.npy', centroids)

    index = indexer.build_faiss_index(embeddings)
    faiss.write_index(index, "models/faiss_index.bin")

    print(index)
    