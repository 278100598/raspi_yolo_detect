From  nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /yolov7
COPY . .
#WORKDIR "/yolov7"
RUN pip install -r requirements.txt
#RUN python python detect.py --weights ./yolov7.pt --conf 0.25 --img-size 640 --source /inference/images/horses.jpg
#RUN python3 new_main.py