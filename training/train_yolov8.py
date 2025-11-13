# train_yolov8_football.py

from ultralytics import YOLO

# Path to your custom dataset YAML
dataset_yaml = "dataset/data.yaml"

# Load YOLOv8 model (pretrained on COCO for transfer learning)
model = YOLO("yolov8s.pt")  # or yolov8n.pt for a smaller model

# Train model on your dataset for 20 epochs (adjust as needed)
results = model.train(
    data=dataset_yaml,  # points to dataset config
    epochs=20,
    imgsz=640,
    batch=16,
    project="football_yolo_training",
    name="exp",
    exist_ok=True,
)

# Save trained model as best.pt
model.save("best.pt")

print("Finished training. Model saved as best.pt.")
