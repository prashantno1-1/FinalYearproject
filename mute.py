import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Global variable to store mute state
is_muted = False

# Function to recognize gestures for mute/unmute
def recognize_gesture(landmarks):
    # Get the landmarks for each finger tip
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12]  # Middle finger tip
    ring_tip = landmarks[16]  # Ring finger tip
    pinky_tip = landmarks[20]  # Pinky tip

    # Check for Fist Gesture (all fingers curled)
    # We compare the y-coordinate of the fingertip with that of the base of the finger
    if thumb_tip.y < landmarks[3].y and index_tip.y < landmarks[7].y and middle_tip.y < landmarks[11].y and ring_tip.y < landmarks[15].y and pinky_tip.y < landmarks[19].y:
        print("Fist gesture detected.")
        return "Fist"
    
    # Check for Open Hand Gesture (fingers extended)
    elif thumb_tip.y > landmarks[3].y and index_tip.y > landmarks[7].y and middle_tip.y > landmarks[11].y and ring_tip.y > landmarks[15].y and pinky_tip.y > landmarks[19].y:
        print("Open hand gesture detected.")
        return "Open Hand"
    
    return "Unknown Gesture"

# Function to mute or unmute based on gesture
def mute_unmute():
    global is_muted
    if is_muted:
        pyautogui.hotkey('ctrl', 'm')  # Send mute/unmute keyboard shortcut (Ctrl + M)
        is_muted = False
        print("Unmuted")
    else:
        pyautogui.hotkey('ctrl', 'm')  # Send mute/unmute keyboard shortcut (Ctrl + M)
        is_muted = True
        print("Muted")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # If hands are found, draw landmarks and recognize gesture
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Recognize gesture based on landmarks
            gesture = recognize_gesture(landmarks)

            # Display the recognized gesture on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # If gesture is "Fist", mute/unmute
            if gesture == "Fist":
                mute_unmute()

    # Display the resulting frame
    cv2.imshow('Gesture Control - Mute/Unmute', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
