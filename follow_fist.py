import cv2
import socket
import depthai as dai
import mediapipe as mp
from cvzone.PoseModule import PoseDetector

# TCP Settings
CONTROLLER_IP = "100.87.161.11"
CONTROLLER_PORT = 9999

# Setup socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((CONTROLLER_IP, CONTROLLER_PORT))
print(f"[Follower] Connected to controller at {CONTROLLER_IP}:{CONTROLLER_PORT}")

# Setup DepthAI pipeline for color camera
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Create pose detector
pose_detector = PoseDetector()

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Frame width for center calculations
frame_width = 640
frame_center = frame_width // 2
center_tolerance = frame_width // 10  # Tolerance around center

def is_fist(hand_landmarks):
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18), (4, 3)]
    count_closed = 0
    for tip, lower in fingers:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[lower].y:
            count_closed += 1
    return count_closed >= 4  # Consider fist if 4+ fingers curled

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            # Flip for natural movement
            frame = cv2.flip(frame, 1)

            # Detect pose
            img = pose_detector.findPose(frame)
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

            # Movement decision
            if fist_detected:
                command = 'x'
            elif bboxInfo is not None and 'bbox' in bboxInfo:
                x, y, w, h = bboxInfo['bbox']
                cx = x + w // 2
                offset = cx - frame_center
                if abs(offset) < center_tolerance:
                    command = 'w'
                elif offset < 0:
                    command = 'a'
                else:
                    command = 'd'
                # Draw tracking box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(img, (cx, y + h // 2), 5, (0, 0, 255), cv2.FILLED)
            else:
                command = 'x'

            # Send TCP command
            try:
                client_socket.sendall(command.encode())
                print(f"[Follower] Sent command: {command}")
            except Exception as e:
                print(f"[Follower][TCP ERROR]: {e}")

            cv2.imshow("Follower View", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())  # Stop before quitting
                break
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Follower] Shutdown complete.")