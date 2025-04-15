import cv2
import numpy as np
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            index_finger = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            x = int(index_finger.x * screen_width)
            y = int(index_finger.y * screen_height)
            
            pyautogui.moveTo(x, y)
            
            pinch_distance = np.linalg.norm(
                np.array([index_finger.x, index_finger.y]) - np.array([thumb.x, thumb.y])
            )
            
            if pinch_distance < 0.05:
                pyautogui.click()
                cv2.putText(frame, "Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            middle_pinch_distance = np.linalg.norm(
                np.array([middle_finger.x, middle_finger.y]) - np.array([thumb.x, thumb.y])
            )
            
            if middle_pinch_distance < 0.05:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
