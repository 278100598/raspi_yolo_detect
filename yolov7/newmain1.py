import random

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, apply_classifier
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier

#added
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from numpy import asarray
import socket


device = 'cuda'
model = attempt_load('./yolov7.pt', map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(640, s=stride)
model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#df = LoadImages('../../class.jpg', img_size=imgsz, stride=stride)
#ret = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))


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

    people = 0
    cnt = [[0, 0], [0, 0]]

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            if names[int(cls)] == 'person':
                if xyxy[0].item() + xyxy[2].item() >= im0.shape[1]:
                    if xyxy[1].item() + xyxy[3].item() >= im0.shape[0]:
                        cnt[1][1] += 1
                    else:
                        cnt[0][1] += 1
                else:
                    if xyxy[1].item() + xyxy[3].item() >= im0.shape[0]:
                        cnt[1][0] += 1
                    else:
                        cnt[0][0] += 1
                people += 1

    #cv2.imshow('Title',im0)
    return ' '.join([str(people),str(cnt[0][0]),str(cnt[0][1]),str(cnt[1][0]),str(cnt[1][1])])


#cap = cv2.VideoCapture(0)
#while (True):
#   # Capture frame-by-frame
#    ret, frame = cap.read()
#
#   cnt = update(frame)
#    print(cnt)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

HOST = '0.0.0.0'
PORT = 12345
img = 'in.jpg'

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)

def go():
    #print('server start at: %s:%s' % (HOST, PORT))
    #print('wait for connection...')

    con, addr = s.accept()
    #print('connected by ' + str(addr))

    with open('in.jpg','wb') as f:
        while True:
            indata = con.recv(1024)
            f.write(indata)
            if len(indata) == 0:
                con.close()
                break

    con, addr = s.accept()
    img = Image.open('in.jpg')
    frame = asarray(img)
    cnt = update(frame)
    #print(cnt)
    con.send(cnt.encode())
    con.close()

while True:
    go()
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

s.close()