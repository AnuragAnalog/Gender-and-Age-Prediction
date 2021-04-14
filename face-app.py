#!/usr/bin/python3

import sys
import cv2
import flask
import numpy as np
import tensorflow as tf
from flask import request, render_template, Response

app = flask.Flask(__name__)
app.config['DEBUG'] = True

model_dir = './models/'
age_labels = {0: 'Male', 1: 'Female'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/video_feed')
def video_feed():
    video = cv2.VideoCapture(0)

    model_arch = 'vgg16'
 
    model = create_model(model_arch)
    loaded_model = load_model(model, model_dir+f'model_weights_{model_arch}.hdf5')
 
    return Response(gen_video(loaded_model, video), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_video(loaded_model, cam):
    clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
    if not cam.isOpened():
        print("Cannot open camera exiting......")
        sys.exit()
 
    while True:
        success, frame = cam.read()
 
        if not success:
            print("Can't receive frame. Exiting ...")
            break
 
        frame = cv2.flip(frame, 1, 1)
        faces = clf.detectMultiScale(frame)
 
        for face in faces:
            (x, y, w, h) = [v for v in face]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48)) / 255.
 
            result = loaded_model.predict(np.expand_dims(face_img, axis=0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0))
            label = f'Age is {np.squeeze(result[0]).round()} and Gender is {age_labels[np.squeeze(result[1]).round()]}'
            cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
 
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def create_model(model_arch):
    model = tf.keras.models.load_model(model_dir+f'model_{model_arch}.keras')

    return model

def load_model(model, fname):
    model.load_weights(fname)

    return model

if __name__ == '__main__':
    app.run()