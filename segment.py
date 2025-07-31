import torch
import torchvision.transforms as T
import cv2
import numpy as np

class SegmentationModel:
    def __init__(self, model_path='resnet_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        # Define preprocessing transformations matching your training
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),    # or your model's input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet means
                        std=[0.229, 0.224, 0.225])
        ])

    def run_inference(self, frame):
        # Preprocess frame
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)  # forward pass

        # Assuming output is logits for segmentation mask classes
        # Convert to probabilities (e.g. softmax) and get mask class 1 probability
        probs = torch.softmax(output, dim=1)
        mask = probs[0,1,...].cpu().numpy()  # class 1 mask probability

        # Resize mask back to original frame size
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Threshold mask to binary (you may want to tune threshold)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        return binary_mask