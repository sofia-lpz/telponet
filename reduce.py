from ultralytics import YOLO

model=YOLO("best.pt")


model.export(format="openvino", imgsz=640, int8= True)  # export to OpenVINO