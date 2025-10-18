from sentence_transformers import SentenceTransformer
import numpy as np
import json

def fetch_unique_templates(segments_json="models/training_segments.json"):
    templates = []
    with open(segments_json, "r") as f:
        segments = json.load(f)
        temp_id_set = set([seg['template_id'] for seg in segments])
        for seg in reversed(segments):
            if seg['template_id'] in temp_id_set:
                templates.append(seg['template'])
                temp_id_set.remove(seg['template_id'])
    return templates

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        Accepts a list of log templates or segments.
        Returns a NumPy array of embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

if __name__ == "__main__":
    
    segments_json = "models/training_segments.json"
    
    templates = fetch_unique_templates(segments_json)
    print(f"Unique templates count: {len(templates)}")

    embedder = Embedder()
    embeddings = embedder.embed(templates)

    with open("models/template_embeddings.npy", "wb") as f:
        np.save(f, embeddings)

    print("Embedding shape:", embeddings.shape)
    print("First vector:", embeddings[0][:5])