from ultralytics import YOLO

# # Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/home/c515/lmx/ultralytics-8.3.35/temp_datasets/temp2/data.yaml", 
                      epochs=1000,
                      imgsz=1920,
                      batch=6,
                      device=0)

# # Evaluate the model's

# # Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# # Export the model to ONNX format
# success = model.export(format="onnx") performance on the validation set
# results = model.val()