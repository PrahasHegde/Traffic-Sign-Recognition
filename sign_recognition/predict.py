# Traffic Sign Recognition Prediction Script

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import cv2
import numpy as np
from collections import deque, Counter

# Model Architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TrafficResNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Predictor Class with Full Dictionary
class TrafficSignPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Full GTSRB Class Mapping
        self.class_names = {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
            3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
            6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
            9: 'No passing', 10: 'No passing veh over 3.5t', 11: 'Right-of-way at intersection', 
            12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
            16: 'Veh > 3.5t prohibited', 17: 'No entry', 18: 'General caution', 
            19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
            22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
            25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
            29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
            32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
            35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
            38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 
            41: 'End of no passing', 42: 'End no passing veh > 3.5t'
        }
        
        # Initialize and Load Model
        self.model = TrafficResNet(num_classes=43).to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found at {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Exact transform used in training
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

    def predict(self, pil_image):
        img_t = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()

# Stability Filter and Camera Loop
def run_camera(predictor):
    cap = cv2.VideoCapture(0)
    # deque stores the last 10 predictions to smooth the output
    prediction_buffer = deque(maxlen=10)
    
    print("Camera active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Pre-process for model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        class_id, conf = predictor.predict(pil_img)
        
        # Only add to buffer if confidence is high enough
        if conf > 0.5:
            prediction_buffer.append(class_id)
        
        # Get the most common prediction in the buffer (Voting)
        if prediction_buffer:
            most_common = Counter(prediction_buffer).most_common(1)[0][0]
            label = predictor.class_names[most_common]
            display_text = f"{label} ({conf*100:.1f}%)"
        else:
            display_text = "Searching for signs..."

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1) # Black header bar
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Real-time TSR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Entry Point
if __name__ == "__main__":
    # Ensure your weight file matches this path
    WEIGHTS = 'models/traffic_sign_model.pth'
    
    try:
        tsr = TrafficSignPredictor(WEIGHTS)
        
        print("1: Static Image Mode")
        print("2: Real-time Camera Mode")
        choice = input("Select mode: ")

        if choice == '1':
            path = input("Enter image path: ")
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                idx, prob = tsr.predict(image)
                print(f"Detected: {tsr.class_names[idx]} | Confidence: {prob*100:.2f}%")
                image.show()
            else:
                print("Invalid path.")
        elif choice == '2':
            run_camera(tsr)
            
    except Exception as e:
        print(f"Error: {e}")