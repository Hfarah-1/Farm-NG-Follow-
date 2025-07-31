import cv2
from cvzone.PoseModule import PoseDetector

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam
cap.set(3, 640)
cap.set(4, 480)

# Create pose detector
detector = PoseDetector()

# Frame dimensions for movement decision logic
frame_width = 640
frame_center = frame_width // 2
center_tolerance = frame_width // 10  # 10% tolerance around center

print("[LocalFollower] Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("[Error] Unable to access webcam.")
        break

    # Detect pose
    img = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)

    if bboxInfo is not None and 'bbox' in bboxInfo:
        x, y, w, h = bboxInfo['bbox']
        cx = x + w // 2  # center x of bbox

        # Draw bounding box and center dot
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)

        # Movement decision based on bounding box center offset
        offset = cx - frame_center
        if abs(offset) < center_tolerance:
            command = 'w'  # Forward
        elif offset < 0:
            command = 'a'  # Turn left
        else:
            command = 'd'  # Turn right
    else:
        command = 'x'  # Stop / no person

    print(f"[LocalFollower] Command: {command}")

    # Display image
    cv2.imshow("Local Follower View", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[LocalFollower] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()