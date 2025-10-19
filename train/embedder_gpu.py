from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time

def fetch_templates(segments_json="models/training_segments.json"):
    templates = []
    with open(segments_json, "r") as f:
        segments = json.load(f)
    templates = [seg["template"] for seg in segments if "template" in seg]
    return templates


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        Accepts a list of log templates or segments.
        Returns a NumPy array of embeddings.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

if __name__ == "__main__":

    segments_json = "models/training_segments.json" 

    start_time = time.time()
    print(f"Start time: {start_time:.3f} seconds")

    templates = fetch_templates(segments_json)
    print(f"Templates count: {len(templates)}")

    templates = [t for t in templates if isinstance(t, str)]

    embedder = Embedder()
    embeddings = embedder.embed(templates)

    with open("models/template_embeddings.npy", "wb") as f:
        np.save(f, embeddings)

    print("Embedding shape:", embeddings.shape)
    print("First vector:", embeddings[0][:5])

    end_time = time.time()
    print(f"End time: {end_time} seconds")

    print(f"Execution time: {end_time - start_time:.3f} seconds")