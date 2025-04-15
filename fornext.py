import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,                   # Using one hand for gesture detection
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open the webcam.
cap = cv2.VideoCapture(0)

# Variable to store the previous x-coordinate of the index finger tip.
prev_x = None

# Define a threshold in pixels to consider as a valid gesture.
gesture_threshold = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect and convert BGR to RGB.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hand landmarks.
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Retrieve image dimensions.
            h, w, _ = frame.shape
            
            # Get the index finger tip landmark (using MediaPipe enum).
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            
            # Draw a circle on the index finger tip for visualization.
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            
            # Set up the initial previous coordinate if it's None.
            if prev_x is None:
                prev_x = cx
            
            # Calculate horizontal movement.
            diff = cx - prev_x
            
           # Check for a forward (right swipe) gesture.
            if diff > gesture_threshold:
                cv2.putText(frame, "Forward Gesture Detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Forward Gesture Detected")
                pyautogui.press('right')  # 👉 Simulate right arrow key
                prev_x = cx

            # Check for a previous (left swipe) gesture.
            elif diff < -gesture_threshold:
                cv2.putText(frame, "Previous Gesture Detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Previous Gesture Detected")
                pyautogui.press('left')  # 👉 Simulate left arrow key
                prev_x = cx

            
            # Optional: update prev_x gradually to track smooth movement.
            # prev_x = int(0.8 * prev_x + 0.2 * cx)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
