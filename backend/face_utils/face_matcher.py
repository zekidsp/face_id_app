import numpy as np
import os
import json

DATA_DIR = "backend/known_faces"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

os.makedirs(DATA_DIR, exist_ok=True)


def load_database():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(METADATA_PATH):
        return np.empty((0, 128)), []  # assuming FaceNet embeddings (128D)

    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return embeddings, metadata


def save_database(embeddings, metadata):
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def find_best_match(embedding, embeddings, threshold=0.7):
    """
    Compare new embedding with stored embeddings using cosine similarity.
    Returns best match index and score.
    """
    if len(embeddings) == 0:
        return None, 0.0

    # Normalize embeddings to unit vectors
    norm_db = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_query = embedding / np.linalg.norm(embedding)

    similarities = np.dot(norm_db, norm_query)
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score >= threshold:
        return best_idx, best_score
    else:
        return None, best_score


def enroll_new_user(embedding, name, metadata):
    """
    Add a new user's embedding and metadata to the database.
    """
    embeddings, meta = load_database()

    if len(embeddings) == 0:
        embeddings = np.expand_dims(embedding, axis=0)
    else:
        embeddings = np.vstack([embeddings, embedding])

    meta.append({"name": name})
    save_database(embeddings, meta)
