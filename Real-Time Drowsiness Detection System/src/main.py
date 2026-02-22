import cv2
import mediapipe as mp
from ear import calculate_EAR, LEFT_EYE, RIGHT_EYE
from utils import get_eye_landmarks
from alarm import play_alarm, stop_alarm
import time


EAR_THRESHOLD = 0.10
FRAME_THRESHOLD = 50

counter = 0
start_time = time.time()
frame_count = 0

blink_interval = 0.5
last_blink_time = time.time()
show_alert = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape

        for face_landmarks in results.multi_face_landmarks:
            left_eye = get_eye_landmarks(face_landmarks, LEFT_EYE, w, h)
            right_eye = get_eye_landmarks(face_landmarks, RIGHT_EYE, w, h)

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            ear = (left_ear + right_ear) / 2.0

            
            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

            # Drowsiness Logic
            if ear < EAR_THRESHOLD:
                counter += 1

                cv2.putText(frame,
                            f"Closed Frames: {counter}",
                            (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2)

                if counter >= FRAME_THRESHOLD:

                    current_time = time.time()

                    # Toggle blinking
                    if current_time - last_blink_time > blink_interval:
                        show_alert = not show_alert
                        last_blink_time = current_time

                    if show_alert:
                        # Red flash overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0),
                                      (frame.shape[1], frame.shape[0]),
                                      (0, 0, 255), -1)

                        alpha = 0.15
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                        cv2.putText(frame,
                                    "DROWSINESS ALERT!",
                                    (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 0, 255),
                                    4)

                    play_alarm()
            else:
                counter = 0
                stop_alarm()
    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
