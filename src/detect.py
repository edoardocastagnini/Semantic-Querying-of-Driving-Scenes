from ultralytics import YOLO

def load_detector(model_name: str = "yolo26x.pt"):
    return YOLO(model_name)