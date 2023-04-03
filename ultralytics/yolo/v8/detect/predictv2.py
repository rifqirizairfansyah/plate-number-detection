# Ultralytics YOLO 🚀, GPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from pymongo import MongoClient

import easyocr
import cv2
import numpy as np
import re

area=[(0, 268), (680, 268), (680, 430), (0, 430)]
area_c=set()


reader = easyocr.Reader(['id'], gpu=True)
client = MongoClient('mongodb+srv://admin:admin@developmentdatabase.6twbbjd.mongodb.net/test',27017)

db = client['test']
plate_number_collection = db['users']

def validate_license_plate(license_plate):
    # Check that the license plate is in the correct format
    if not re.match(r'^[A-Z]{1,2}\s[0-9]{3,4}\s[A-Z]{2,3}$', license_plate):
        return False

    findExistPlateNumber = plate_number_collection.find_one({ 'plate_number': license_plate })
    # Check that the license plate does not already exist in the database
    if findExistPlateNumber:
        return True
        
    # Add additional checks if needed

    return False

def ocr_image(img,coordinates):
    x,y,w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])

    cx=int(x+w)//2
    cy=int(y+h)//2
    # cv2.circle(img, (cx, cy), 3, (255,0,0), 15)
    img = img[y:h,x:w]
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    # inDetectionPlane = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
    gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 3)
    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    
    text = ""

    # if inDetectionPlane>=0:
    result = reader.readtext(gray)
    cv2.imshow("Otsu", thresh)

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
    #     text += res[1] + " "
    if validate_license_plate(text):
        print('exist')
        plate_number_collection.insert_one({ 'plate_number': text })
    
    # plate_number_collection.find_one({ 'plate_number': text })
    print(text)
    return str(text)

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return f'{log_string}(no detections), '
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            if self.args.save_txt:  # Write to file
                line = (c, *d.xywhn.view(-1)) + (conf, ) * self.args.save_conf + (() if id is None else (id, ))
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.show:  # Add bbox to image
                name = ('' if id is None else f'id:{id} ') + self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                text_ocr = ocr_image(im0, d.xyxy[0])
                label = text_ocr
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or '1000_image_best.pt'
    print(cfg.source)
    source = 'demo.mp4'

    args = dict(model=model, source=source, show=True, save_txt=True, save_crop=True)
    print(args)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()