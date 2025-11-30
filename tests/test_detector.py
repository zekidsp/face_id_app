import numpy as np
from backend.face_utils.face_detector import detect_and_crop_face

def test_detect_face_on_blank_image():
    blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cropped = detect_and_crop_face(blank.tobytes())
    assert cropped is None
