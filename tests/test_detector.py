import numpy as np
from backend.face_utils.face_detector import detect_and_crop_face
import cv2

def test_detect_face_on_blank_image():
    blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", blank)
    cropped = detect_and_crop_face(buf.tobytes())
    assert cropped is None
