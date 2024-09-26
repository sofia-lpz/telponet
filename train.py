from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="/home/sofia/Documents/telponet/alt_dataset/data.yaml", epochs=2000, patience=100)  # train the model

model.export(format="openvino", imgsz=640)  # export to OpenVINO
