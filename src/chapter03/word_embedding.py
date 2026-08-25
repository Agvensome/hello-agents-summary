# word_embedding.py

import numpy as np

embeddings = {
    "king": np.array([.9, .8]),
    "man": np.array([.7, .9]),
    "queen": np.array([.9, .2]),
    "woman": np.array([.7, .3])
}

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product

# king - man + woman
res_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]

# calculate similarity
sim = cosine_similarity(res_vec, embeddings["queen"])

print(f"king - man + woman: {res_vec}")
print(f"The simlarity with 'queen': {sim:.4f}")