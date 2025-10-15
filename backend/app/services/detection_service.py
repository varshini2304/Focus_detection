import io
from typing import List, Tuple, Dict, Any
import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # allow backend to start without OpenCV
import asyncio

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception:  # pragma: no cover
    mp_face_mesh = None

_yolo_model = None
_face_mesh = None


async def load_models():
    global _yolo_model, _face_mesh
    loop = asyncio.get_running_loop()

    async def _load_yolo():
        global _yolo_model
        if YOLO is not None:
            try:
                _yolo_model = await loop.run_in_executor(None, YOLO, "yolov8n.pt")
            except Exception:
                # Fallback gracefully if weights cannot be downloaded/loaded
                _yolo_model = None

    async def _load_facemesh():
        global _face_mesh
        if mp_face_mesh is not None:
            try:
                # Static image mode False for video stream, 1 face, refine landmarks, reasonable confidence
                _face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
            except Exception:
                _face_mesh = None

    await asyncio.gather(_load_yolo(), _load_facemesh())


def _bytes_to_bgr(img_bytes: bytes) -> np.ndarray | None:
    if cv2 is None:
        return None
    data = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return bgr


def _compute_ear(landmarks: List[tuple]) -> float:
    # Eye Aspect Ratio based on 6 points per eye (approx using FaceMesh indices)
    # Using standard indices for left eye: [33, 160, 158, 133, 153, 144]
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    left = [landmarks[i] for i in [33, 160, 158, 133, 153, 144] if i < len(landmarks)]
    right = [landmarks[i] for i in [263, 387, 385, 362, 380, 373] if i < len(landmarks)]
    def ear(eye):
        if len(eye) < 6:
            return 0.3
        A = dist(eye[1], eye[5])
        B = dist(eye[2], eye[4])
        C = dist(eye[0], eye[3])
        return (A + B) / (2.0 * C + 1e-6)
    return float((ear(left) + ear(right)) / 2.0)


def _compute_mor(landmarks: List[tuple]) -> float:
    # Mouth Opening Ratio using landmark pairs
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    # Approximate inner mouth indices
    mouth = [landmarks[i] for i in [13, 14, 78, 308] if i < len(landmarks)]
    if len(mouth) < 4:
        return 0.2
    vertical = dist(mouth[0], mouth[1])
    horizontal = dist(mouth[2], mouth[3])
    return float(vertical / (horizontal + 1e-6))


def _head_pose_yaw(landmarks: List[tuple]) -> float:
    # Very rough proxy using nose vs eyes to estimate yaw angle in degrees
    # Return absolute yaw angle
    try:
        nose = np.array(landmarks[1])  # approximation: landmark 1 near nose bridge
        left_eye = np.array(landmarks[33])
        right_eye = np.array(landmarks[263])
    except Exception:
        return 0.0
    mid_eye = (left_eye + right_eye) / 2.0
    vec = nose - mid_eye
    # Angle relative to camera forward (heuristic)
    angle = np.degrees(np.arctan2(vec[0], vec[1] + 1e-6))
    return float(abs(angle))


async def process_frame(img_bytes: bytes) -> Tuple[List[str], Dict[str, Any]]:
    bgr = _bytes_to_bgr(img_bytes)
    events: List[str] = []
    features: Dict[str, Any] = {}

    # YOLO object detection (phone)
    phone_detected = False
    if _yolo_model is not None and bgr is not None:
        results = await asyncio.get_running_loop().run_in_executor(None, _yolo_model, bgr)
        try:
            # Ultralytics results: take first result boxes
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                # COCO class 67 is cell phone
                if cls_id == 67 and float(box.conf[0]) > 0.3:
                    phone_detected = True
                    break
        except Exception:
            phone_detected = False
    features["phone_flag"] = phone_detected
    if phone_detected:
        events.append("phone")

    # Face landmarks
    landmarks_xy: List[tuple] = []
    if _face_mesh is not None and bgr is not None and cv2 is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fm_res = _face_mesh.process(rgb)
        if fm_res.multi_face_landmarks:
            h, w = bgr.shape[:2]
            lm = fm_res.multi_face_landmarks[0]
            landmarks_xy = [(p.x * w, p.y * h) for p in lm.landmark]

    ear = _compute_ear(landmarks_xy) if landmarks_xy else 0.3
    mor = _compute_mor(landmarks_xy) if landmarks_xy else 0.2
    yaw = _head_pose_yaw(landmarks_xy) if landmarks_xy else 0.0

    features.update({"ear": ear, "mor": mor, "yaw": yaw})

    # Heuristic events
    if ear < 0.18:
        events.append("eyes_closed")
    if mor > 0.6:
        events.append("yawn")
    if yaw > 30.0:
        events.append("looking_away")

    # Placeholder smile/talk detection (heuristic)
    if 0.35 < mor < 0.6:
        events.append("talking")

    return events, features


async def health() -> Dict[str, bool]:
    return {
        "yolo": _yolo_model is not None,
        "faceMesh": _face_mesh is not None,
    }
