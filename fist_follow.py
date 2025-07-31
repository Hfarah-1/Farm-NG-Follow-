import cv2
import socket
import depthai as dai
import mediapipe as mp

# TCP Settings
CONTROLLER_IP = "100.87.161.11"
CONTROLLER_PORT = 9999

# Setup TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((CONTROLLER_IP, CONTROLLER_PORT))
print(f"[FistFollower] Connected to controller at {CONTROLLER_IP}:{CONTROLLER_PORT}")

# Setup DepthAI color camera pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Helper: check if hand is a fist
def is_fist(landmarks):
    # Tip.y > PIP.y means finger is curled
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip(tips, pips))

# Main loop
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)
            command = 'x'  # Default: stop

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_fist(hand_landmarks.landmark):
                        command = 'w'
                    else:
                        command = 'x'
                    break  # Only check first hand

            try:
                client_socket.sendall(command.encode())
                print(f"[FistFollower] Sent: {command}")
            except Exception as e:
                print(f"[TCP ERROR]: {e}")
                break

            cv2.imshow("Fist Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('x'.encode())  # Stop before exit
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[FistFollower] Shutdown complete.")