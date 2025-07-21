import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Start webcam
cap = cv2.VideoCapture(0)

# Focus score storage
focus_scores = []
start_time = time.time()
duration_minutes = 30  # You can change this to 1 for 1 hour

def calculate_focus_score(landmarks, image_w, image_h):
    def to_px(pt): return np.array([int(pt.x * image_w), int(pt.y * image_h)])

    l_eye = to_px(landmarks[33])
    r_eye = to_px(landmarks[263])
    nose = to_px(landmarks[1])
    l_ear = to_px(landmarks[234])
    r_ear = to_px(landmarks[454])
    top_lid = to_px(landmarks[159])
    bottom_lid = to_px(landmarks[145])
    mouth_upper = to_px(landmarks[13])
    mouth_lower = to_px(landmarks[14])

    eye_dist = np.linalg.norm(l_eye - r_eye)
    head_tilt = abs(l_ear[1] - r_ear[1])
    nose_offset = abs((l_eye[0] + r_eye[0]) / 2 - nose[0])
    eye_open = np.linalg.norm(top_lid - bottom_lid)
    mouth_open = np.linalg.norm(mouth_upper - mouth_lower)

    focus = 100
    if nose_offset > 40: focus -= 20
    if head_tilt > 30: focus -= 20
    if eye_dist < 60: focus -= 10
    if eye_open < 5: focus -= 20  # Eye closed
    if mouth_open > 15: focus -= 20  # Talking

    return max(0, min(100, focus))

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Couldn't access webcam.")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            focus = calculate_focus_score(face_landmarks.landmark, w, h)
            focus_scores.append(focus)

            # Display current focus % (instant)
            label = f"Focus Now: {int(focus)}%"
            color = (0, 255, 0) if focus >= 70 else (0, 0, 255)
            cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display average focus % (overall)
            if focus_scores:
                avg_focus = sum(focus_scores) / len(focus_scores)
                avg_label = f"Total Focus: {int(avg_focus)}%"
                cv2.putText(frame, avg_label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Timer display
    elapsed_min = (time.time() - start_time) / 60
    cv2.putText(frame, f"Time: {int(elapsed_min)} min", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # End session after given duration
    if elapsed_min >= duration_minutes:
        avg_focus = sum(focus_scores) / len(focus_scores)
        print(f"\n✅ Session Complete: Total Average Focus = {avg_focus:.2f}%\n")
        break

    cv2.imshow("Student Focus Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
