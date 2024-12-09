import torch
from ultralytics import YOLO

# Select device
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'african-wildlife.yaml' dataset for 3 epochs
results = model.train(data="african-wildlife.yaml", epochs=3, imgsz=640)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/assets/african-wildlife-sample.jpg")
print(results)