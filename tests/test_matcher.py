from backend.face_utils.face_matcher import find_best_match
import numpy as np

def test_matcher_closest_match():
    embeddings = np.array([
        np.zeros(128),
        np.ones(128),
    ])
    target = np.ones(128)
    
    idx, score = find_best_match(target, embeddings)
    assert idx == 1
