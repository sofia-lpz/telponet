from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="/home/sofia/Documents/telponet/alt_dataset/data.yaml", 
            epochs=3000, 
            patience=150,
            batch=0.70,
            name = "nyu_hyper1",
            shear = 90,
            )  # train the model

model.export(format="openvino", imgsz=640)  # export to OpenVINO
