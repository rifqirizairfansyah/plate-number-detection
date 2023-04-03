from ultralytics import YOLO
model = YOLO("E:/Project 2023/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/1000_image_best.pt")
model.predict(source=0, conf=0.5)