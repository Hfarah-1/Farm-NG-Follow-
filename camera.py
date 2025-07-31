import depthai as dai
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp  # UNet with ResNet34 backbone

class OAKCamera:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    def get_frame(self):
        in_rgb = self.q_rgb.tryGet()
        if in_rgb is None:
            return None
        return in_rgb.getCvFrame()

    def close(self):
        self.device.close()

class SegmentationModel:
    def __init__(self, model_path='unet_resnet34Final.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create UNet model with ResNet34 encoder
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((320, 320)),  # Adjust to your training size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def run_inference(self, frame):
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            # output shape: (batch_size, classes=1, H, W)
            mask_prob = torch.sigmoid(output)[0, 0].cpu().numpy()

        mask_resized = cv2.resize(mask_prob, (frame.shape[1], frame.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        return binary_mask

def overlay_mask(frame, mask, color=(0, 0, 255), alpha=0.5):
    color_mask = np.zeros_like(frame)
    color_mask[mask == 1] = color
    overlayed = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlayed

def main():
    cam = OAKCamera()
    model = SegmentationModel()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            mask = model.run_inference(frame)
            overlayed = overlay_mask(frame, mask)

            cv2.imshow("OAK Camera with Segmentation Overlay", overlayed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()