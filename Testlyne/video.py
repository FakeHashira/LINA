import cv2
from ultralytics import YOLO


video_path = './Testlyne/image/animals.mp4'

# # load the video
video = cv2.VideoCapture(video_path)

model = YOLO('yolov8n.pt')

results = model(source='video', show=True, conf=0.4, save=True)

