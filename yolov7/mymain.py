import random

import cv2
import numpy as np
import torch

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, apply_classifier
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import load_classifier

device = 'cuda'
model = attempt_load('yolov7.pt', map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(640, s=stride)
model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

df = LoadImages('../class.jpg', img_size=imgsz, stride=stride)
ret = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))


def update(im0):
    # Padded resize
    img = letterbox(im0, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]
    det = non_max_suppression(pred)[0]


    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        cnt = 0
        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            if names[int(cls)] == 'person':
                cnt += 1
        cv2.imshow('Title',im0)
        return cnt


cap = cv2.VideoCapture(0)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cnt = update(frame)
    print(cnt)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break