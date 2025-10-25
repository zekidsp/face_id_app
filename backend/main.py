from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from backend.face_utils.face_detector import detect_and_crop_face
from backend.face_utils.face_embedder import extract_face_embedding
import cv2
import numpy as np
import os

app = FastAPI(title="Face Identification API", version="0.3")

DETECTED_DIR = "backend/detected_faces"
os.makedirs(DETECTED_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Face Identification API is running"}

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Step 1: Detect and crop
    cropped_face = detect_and_crop_face(image_bytes)
    if cropped_face is None:
        return JSONResponse({"status": "no_face_detected"})
    
    # Step 2: Save cropped face (for debugging)
    save_path = os.path.join(DETECTED_DIR, f"cropped_{file.filename}")
    cv2.imwrite(save_path, cropped_face)
    
    # Step 3: Extract embedding
    embedding = extract_face_embedding(cropped_face)
    if embedding is None:
        return JSONResponse({"status": "embedding_failed"})
    
    # Step 4: Return embedding info (for now)
    return JSONResponse({
        "status": "embedding_generated",
        "embedding_length": len(embedding),
        "embedding_sample": embedding[:5].tolist()  # show only first 5 values
    })
