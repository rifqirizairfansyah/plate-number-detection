from ultralytics import YOLO
from ultralytics.yolo.v8.detect import DetectionPredictor

import cv2

model = YOLO("1000_image_best.pt")

result = model.predict(source="0", show=True)
cv2.waitKey()
print(result)

