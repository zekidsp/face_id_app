from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from backend.face_utils.face_detector import detect_and_crop_face
from backend.face_utils.face_embedder import extract_face_embedding
from backend.face_utils.face_matcher import load_database, find_best_match, enroll_new_user
import numpy as np
import os

app = FastAPI(title="Face Identification API", version="0.5")

# Paths
DETECTED_DIR = "backend/detected_faces"
KNOWN_FACES_DIR = "backend/known_faces"
os.makedirs(DETECTED_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)



# Global in-memory cache
embeddings = None
metadata = None

# Preload database at startup for efficiency
@app.on_event("startup")
def load_face_database():
    """Load embeddings and metadata into memory when the app starts"""
    global embeddings, metadata
    embeddings, metadata = load_database()
    print(f"✅ Loaded {len(metadata)} known faces from database.")

@app.get("/")
def root():
    return {"message": "Face Identification API is running"}

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    # Read and preprocess
    image_bytes = await file.read()
    cropped_face = detect_and_crop_face(image_bytes)
    if cropped_face is None:
        return JSONResponse({"status": "no_face_detected"})

    # Extract embedding
    embedding = extract_face_embedding(cropped_face)
    if embedding is None:
        return JSONResponse({"status": "embedding_failed"})

    # Match against loaded database
    global embeddings, metadata
    match_idx, score = find_best_match(embedding, embeddings)

    if match_idx is not None:
        name = metadata[match_idx]["name"]
        return JSONResponse({
            "status": "identified",
            "name": name,
            "similarity_score": float(score)
        })
    else:
        return JSONResponse({
            "status": "unknown",
            "similarity_score": float(score)
        })

@app.post("/enroll")
async def enroll_face(file: UploadFile = File(...), name: str = Form(...)):
    image_bytes = await file.read()
    cropped_face = detect_and_crop_face(image_bytes)
    if cropped_face is None:
        return JSONResponse({"status": "no_face_detected"})

    embedding = extract_face_embedding(cropped_face)
    if embedding is None:
        return JSONResponse({"status": "embedding_failed"})

    # Save new user
    enroll_new_user(embedding, name, metadata=None)

    # Reload in-memory database
    global embeddings, metadata
    embeddings, metadata = load_database()

    return JSONResponse({"status": "enrolled", "name": name})


