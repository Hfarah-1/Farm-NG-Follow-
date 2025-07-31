import cv2
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default Mac webcam

# Initialize detectors
pose_detector = PoseDetector()
hand_detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror view

    # Detect pose
    frame = pose_detector.findPose(frame)
    lmList, bboxInfo = pose_detector.findPosition(frame, bboxWithHands=False)

    # Draw bounding box around pose
    if bboxInfo:
        x, y, w, h = bboxInfo['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, "Pose Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Detect hands
    hands, frame = hand_detector.findHands(frame)

    for hand in hands:
        handType = hand["type"]
        fingers = hand_detector.fingersUp(hand)

        # Display hand type
        x, y, w, h = hand["bbox"]
        label = f"{handType} Hand"

        # Check for fist (all fingers down)
        if fingers == [0, 0, 0, 0, 0]:
            label += " - FIST"
            color = (0, 0, 255)  # Red if fist
        else:
            color = (0, 255, 0)  # Green if open

        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, cv2.FILLED)
        cv2.putText(frame, label, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show final output
    cv2.imshow("Pose and Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()