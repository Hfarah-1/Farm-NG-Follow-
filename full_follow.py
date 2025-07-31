import cv2
import socket
import depthai as dai
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector

# TCP Connection Setup
ROBOT_IP = "100.87.161.11"
PORT = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ROBOT_IP, PORT))
print("[Follower] Connected to robot.")

# Initialize detectors
pose_detector = PoseDetector()
hand_detector = HandDetector(detectionCon=0.8, maxHands=2)

# Create pipeline and camera
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Frame and movement setup
frame_width = 640
frame_center = frame_width // 2
center_tolerance = frame_width // 10  # acceptable range to go straight

# Run on device
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            # Pose detection
            img = pose_detector.findPose(frame)
            lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=True)

            # Hand detection
            hands, img = hand_detector.findHands(img)

            # Check for fist in either hand
            fist_detected = False
            for hand in hands:
                fingers = hand_detector.fingersUp(hand)
                if sum(fingers) == 0:  # All fingers down = fist
                    fist_detected = True
                    break

            if fist_detected:
                command = 'x'
            elif bboxInfo is not None and 'bbox' in bboxInfo:
                x, y, w, h = bboxInfo['bbox']
                cx = x + w // 2
                offset = cx - frame_center

                # Draw bounding box and center dot
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)

                if abs(offset) < center_tolerance:
                    command = 'w'
                elif offset < 0:
                    command = 'a'
                else:
                    command = 'd'
            else:
                command = 'x'

            try:
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")

            # Show the full annotated frame
            cv2.imshow("Follower View", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")

        #test github