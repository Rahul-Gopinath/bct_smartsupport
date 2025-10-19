import numpy as np
import json
import faiss
#import gpt_model

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


class Annotator:
    def __init__(self):
        pass

    def get_representative_logs(self, log_index, centroids, templates, top_k=3):
        D, I = log_index.search(np.array(centroids).astype('float32'), k=top_k)
        print("Indices of representative logs:", I)
        print("Distances of representative logs:", D)
        if top_k == 1:
            print("Length of templates:", len(templates))
            return [templates[i[0]] for i in I]
        return [[templates[i] for i in indices] for indices in I]


# def annotate_clusters(representative_logs):
#     # 2 possible approaches:
#     # 1. Use sentence transformers to find similarity to known categories
#     # 2. Use GPT to classify into known categories

#     annotations = []
#     for log in representative_logs:
#         prompt = f"""You are a log diagnostics assistant. 
#         Generate a concise annotation for the following logs:
#         {log}Return a one-sentence summary of the issue."""
#         response = gpt_model.generate(prompt)  # Assuming gpt_model is defined elsewhere
#         annotations.append(response)
#     return annotations


if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', help = 'model_file', type = str)#, default = './Log2VecOutput/log2vec_words.model')
    # parser.add_argument('--output', help = 'clustered_segments', type = str, default='clustered_segments.json')
    # args = parser.parse_args()

    print("Loading embeddings and labels...")
    log_index = faiss.read_index("models/faiss_index.bin")

    print("Loading centroids...")
    centroids = np.array(list(np.load('models/cluster_centroids.npy', allow_pickle=True).item().values()))

    templates = fetch_unique_templates("models/training_segments.json")

    annotator = Annotator()
    representative_logs = annotator.get_representative_logs(log_index, centroids, templates)

    print("Representative logs for each cluster:")
    for i, log in enumerate(representative_logs):
        print(f"Cluster {i}: {log}...")
    print("\n")
