# Farm-NG-Follow

## Requirements

Make sure you have the following ready:

- Farm-NG Amiga robot powered on and connected to the same network as your laptop.  
- DepthAI camera connected to your laptop via USB.  
- Python 3 installed with required libraries:  
  - `opencv-python`  
  - `depthai`  
  - `cvzone`  
  - `mediapipe`  
  - `torch` and `torchvision` (for `planner_follow.py`)  
  - `segmentation_models_pytorch` (for `planner_follow.py`)  

---

## Robot Side

Open a new terminal on your local machine and SSH into the Farm-NG robot using the following command:

```bash
ssh farm-ng-user-mu-paal@<ip-address>
```
Once connected, navigate to the vehicle twist folder:

```bash
cd ~/Desktop/farm-ng/farm-ng-amiga/py/examples/vehicle_twist/
```
Inside this folder, you will find the controller.py script.

## About `controller.py`

The `controller.py` script runs on the Farm-NG robot and acts as a TCP server. It listens for incoming single-character commands (`w`, `a`, `d`, `x`, etc.) sent from the vision follower scripts running on your laptop. These commands correspond to movement instructions like moving forward, turning left or right, and stopping.

Upon receiving a command, `controller.py` converts it into velocity commands (`Twist2d` messages) which control the Farm-NG Amiga's motors, enabling the robot to follow or react to the detected person or path based on the vision processing happening remotely.

run it with:

```bash
python3 controller.py
```

# Overview of Each Vision Follower Script

## 1. planner.py  
- **Description:** Uses a deep learning segmentation model (UNet ResNet34) to detect the drivable path in camera frames.  
- **Function:** Processes enhanced images to create a binary path mask, then decides whether the robot should move forward, turn left/right, or stop based on the path’s position in the frame.  
- **Features:** Includes image contrast enhancement and live overlay of segmentation mask. This is the foundational script for vision-based autonomous navigation.

## 2. follow.py  
- **Description:** Uses DepthAI camera with cvzone PoseDetector to detect a person and track their horizontal position.  
- **Function:** Commands the robot to move forward, turn left, or turn right based on the bounding box center of the detected person, or stop if no person detected.  
- **Features:** Simple pose-based follower with bounding box visualization.

## 3. height_follow.py  
- **Description:** Builds on `follow.py` by adding distance awareness using the bounding box height.  
- **Function:** Stops the robot when the person is too close (bounding box height exceeds threshold), otherwise moves or turns based on horizontal position.  
- **Features:** Helps avoid collisions using rough distance estimation.

## 4. backtrack_follow.py  
- **Description:** Adds hand detection to pose detection using cvzone HandDetector.  
- **Function:** Detects a hand signal (fist) to permanently stop the robot; otherwise, uses pose bounding box for movement commands.  
- **Features:** Includes bounding box and center dot visualization, with fist stop detection.

## 6. fist_follow.py  
- **Description:** Uses DepthAI camera combined with MediaPipe Hands (not cvzone) for simple hand gesture detection.  
- **Function:** Sends 'w' (move forward) when a fist is detected, otherwise 'x' (stop).  
- **Features:** Lightweight, hand-only control with basic video display.

---
## Running the Vision Follower Scripts (on your local laptop)

After you have the robot-side `controller.py` running via SSH, open a terminal on your local machine, navigate to the folder containing your follower scripts, and run any of the vision follower scripts by using:

```bash
python3 <filename.py>
