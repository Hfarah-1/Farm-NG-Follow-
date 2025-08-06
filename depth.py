import cv2
import depthai as dai
import numpy as np

# Mouse coordinates for clicking
mouseX = 0
mouseY = 0

def getFrame(queue):
    # Get frame from queue and convert to OpenCV format
    frame = queue.get()
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    # Create and configure stereo node
    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # Link mono outputs to stereo inputs
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

if __name__ == "__main__":
    # Create pipeline
    pipeline = dai.Pipeline()

    # Set up mono cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Create stereo depth node
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # XLink outputs
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.disparity.link(xoutDepth.input)

    xoutLeft = pipeline.createXLinkOut()
    xoutLeft.setStreamName("left")
    stereo.rectifiedLeft.link(xoutLeft.input)

    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName("right")
    stereo.rectifiedRight.link(xoutRight.input)

    # Start device
    with dai.Device(pipeline) as device:
        # Output queues
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        # Disparity range 0-95 -> map to 0-255 for visualization
        disparityMultiplier = 255 / stereo.getMaxDisparity()

        sideBySide = True  # toggle key

        cv2.namedWindow("Stereo View")
        cv2.setMouseCallback("Stereo View", mouseCallback)

        while True:
            # Get frames
            disparity = getFrame(qDepth)
            left = getFrame(qLeft)
            right = getFrame(qRight)

            # Normalize disparity for display
            disparity_visual = (disparity * disparityMultiplier).astype(np.uint8)
            disparity_colored = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)

            # Show depth value at clicked point
            cv2.circle(disparity_colored, (mouseX, mouseY), 4, (0, 255, 0), -1)
            value = disparity[mouseY, mouseX]
            cv2.putText(disparity_colored, f"Disparity: {value}", (mouseX + 10, mouseY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if sideBySide:
                stereo_view = np.hstack((left, right))
                cv2.imshow("Stereo View", stereo_view)
            else:
                # Blend left and right views for overlapping stereo visualization
                overlay = cv2.addWeighted(left, 0.5, right, 0.5, 0)
                cv2.imshow("Stereo View", overlay)

            cv2.imshow("Disparity", disparity_colored)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                sideBySide = not sideBySide

        cv2.destroyAllWindows()