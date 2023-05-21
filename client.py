import socket
import time
import cv2
import numpy as np
from PIL import Image as im

server = 'localhost'
port = 12345
cap = cv2.VideoCapture(0)
img = 'photo.jpg'


#img = input('想傳的檔名:')

while True:
    ret, frame = cap.read()
    image = im.fromarray(frame)
    image.save(img)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server, port))
    print("Client start at : ", s.getsockname())
    with open(img, "rb") as f:
        ret = f.read(102400)
        s.send(ret)

    ret = s.recv(1024)
    print(ret.decode())
    s.close()


