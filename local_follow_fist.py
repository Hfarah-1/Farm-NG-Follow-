import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as mp

#Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#Pose detector
pose_detector = PoseDetector()

#MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

#Frame center for logic
frame_width = 640
frame_center = frame_width // 2
center_tolerance = frame_width // 10

print("[LocalFollower] Press 'q' to quit.")

def is_fist(hand_landmarks):
    # Index finger tip = 8, PIP joint = 6
    # Middle = 12/10, Ring = 16/14, Pinky = 20/18, Thumb (simplified): 4/3
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18), (4, 3)]
    count_closed = 0
    for tip, lower in fingers:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[lower].y:
            count_closed += 1
    return count_closed >= 4  # Consider it a fist if at least 4 fingers are curled

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)

    # Pose detection
    img = pose_detector.findPose(frame, draw=True)
    lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=True)

    # Hand detection
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    fist_detected = False
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            if is_fist(handLms):
                fist_detected = True
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Determine movement
    if fist_detected:
        command = 'x'  # Stop if a fist is detected
    elif bboxInfo is not None and 'bbox' in bboxInfo:
        x, y, w, h = bboxInfo['bbox']
        cx = x + w // 2
        offset = cx - frame_center
        if abs(offset) < center_tolerance:
            command = 'w'  # Forward
        elif offset < 0:
            command = 'a'  # Turn left
        else:
            command = 'd'  # Turn right
        # Draw tracking box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)
    else:
        command = 'x'  # Stop if no person is detected

    print(f"[LocalFollower] Command: {command}")

    # Show window
    cv2.imshow("Local Follower with Hand Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[LocalFollower] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()