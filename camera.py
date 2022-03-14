import re
import cv2
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans = human_cascade.detectMultiScale(gray, 1.9, 1)
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            break
        ret, jpeg = cv2.imencode('.jpg',frame)   
        return jpeg.tobytes()  