from ultralytics import YOLO

model=YOLO("drone.pt")


model.export(format="openvino", imgsz=640, int8= True)  # export to OpenVINO