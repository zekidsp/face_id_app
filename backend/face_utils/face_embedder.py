from deepface import DeepFace
import numpy as np
import cv2

def extract_face_embedding(face_img: np.ndarray):
    """
    Generates an embedding vector from a cropped face image.
    Returns a numpy array of embeddings.
    """
    try:
        # DeepFace expects BGR images as input (same as OpenCV)
        embedding_obj = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",     # lightweight and accurate
            enforce_detection=False    # since we already cropped face
        )
        
        # DeepFace returns a list of dicts; extract the embedding
        embedding = np.array(embedding_obj[0]["embedding"])
        return embedding

    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return None
