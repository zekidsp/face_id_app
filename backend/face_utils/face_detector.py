import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO

mp_face_detection = mp.solutions.face_detection

def detect_and_crop_face(image_bytes: bytes):
    """Detects a face in the given image and returns the cropped face region (as np.ndarray)."""
    
    # Convert bytes → NumPy image (OpenCV format)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Initialize Mediapipe Face Detector
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return None
        
        # Use the first detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w, _ = img.shape
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cropped_face = img[y1:y2, x1:x2]
        return cropped_face
