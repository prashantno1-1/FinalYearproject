import cv2
import numpy as np
import mediapipe as mp
import os
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)



cap = cv2.VideoCapture(0)

prev_x = 0  # For swipe gesture tracking

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape

            # Get key points
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            pinky_tip = landmarks[20]

            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
            x3, y3 = int(middle_tip.x * w), int(middle_tip.y * h)
            x4, y4 = int(pinky_tip.x * w), int(pinky_tip.y * h)

            # Calculate distances
            distance_thumb_index = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            distance_thumb_pinky = np.linalg.norm(np.array([x1, y1]) - np.array([x4, y4]))
            distance_index_middle = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))

           

            # 4️⃣ Next/Previous Track (Swipe Gesture)
            current_x = int(landmarks[0].x * w)

            if prev_x - current_x > 50:
                os.system("nircmd.exe sendkeypress next")  # Next track
                cv2.putText(frame, "Next Track", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif current_x - prev_x > 50:
                os.system("nircmd.exe sendkeypress prev")  # Previous track
                cv2.putText(frame, "Previous Track", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

            prev_x = current_x

           

    cv2.imshow("Gesture Volume & Media Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
