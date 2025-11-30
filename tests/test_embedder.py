import numpy as np
from unittest.mock import patch
from backend.face_utils.face_embedder import extract_face_embedding

@patch("face_embedder.DeepFace.represent")
def test_extract_face_embedding_success(mock_represent):
    # Mock DeepFace output
    mock_represent.return_value = [
        {"embedding": [0.1, 0.2, 0.3]}
    ]

    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    embedding = extract_face_embedding(dummy_img)

    assert embedding is not None
    assert embedding.shape == (3,)
    assert embedding.tolist() == [0.1, 0.2, 0.3]

@patch("face_embedder.DeepFace.represent")
def test_extract_face_embedding_failure(mock_represent):
    mock_represent.side_effect = Exception("Model error")

    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    embedding = extract_face_embedding(dummy_img)

    assert embedding is None
