# Farm-NG-Follow-


Follow System Setup Guide (Farm-NG + DepthAI + Pose/Hand Detection Algorithms)
This system uses a DepthAI camera and pose/hand detection to control the movement of the Farm-NG Amiga robot via TCP socket communication.

Prerequisites

Farm-NG Amiga powered on and connected to the same network as your laptop.
DepthAI camera connected to the laptop via USB.
controller.py located in the vehicle_twist/ directory on the Farm-NG.
follow.py on your local laptop.
Python 3 installed (along with OpenCV, MediaPipe, cvzone, etc).


Step-by-Step Instructions

1. Start the Robot-Side TCP Controller on the Farm-NG, open a terminal

   SSH into the Farm-NG:
   ssh farm-ng-user-mu-paal@farm-ng-ip-address

   Navigate to the vehicle twist folder:
   cd ~/Desktop/farm-ng/farm-ng-amiga/py/examples/vehicle_twist/

   Run the controller:
   python3 controller.py

This will start a TCP server socket, which listens commands from the vision system which will be converted into Twist2d velocity commands for the robot to act on.

2. Start the Vision-Side Follower Script
   on your local laptop, open a separate terminal:

   Plug in the DepthAI camera via USB.

   Navigate to the folder containing follow.py:
   cd path/to/your/follow_script_folder/

   Run the script:
   python3 follow.py

   This script uses the DepthAI camera to detect a person using pose detection and computes whether the robot should turn left, right, move forward, or stop. It sends the         corresponding command (w, a, d, x) over the TCP socket to the already-listening controller.py on the robot.

Note: Currently, there is no stop function on the follow.py code.


There is also a follow_fist.py that has implemented a stop function using hand signals (fist). Although it works it hasnt been thoroughly tested yet. To run the follow_fist.py file you would go through all of the same steps, the only difference would be running it using python3 fist_follow.py.Is is made to work with the controller.py that will be running on the robot side 






