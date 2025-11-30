import numpy as np


def find_best_match(target_embedding, embeddings):
    if embeddings is None or len(embeddings) == 0:
        return None, None

    # Normalize target safely
    target_norm = np.linalg.norm(target_embedding)
    if target_norm == 0:
        return None, None
    target_normed = target_embedding / target_norm

    # Normalize database embeddings safely
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6  # avoid division by zero
    embeddings_normed = embeddings / norms

    # Compute cosine similarity
    sims = embeddings_normed @ target_normed

    # If similarity calculation produced NaN, replace with -1
    sims = np.nan_to_num(sims, nan=-1.0)

    idx = np.argmax(sims)
    score = sims[idx]

    return idx, float(score)

