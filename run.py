#!/usr/bin/python3

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def create_model(model_arch):
    model = tf.keras.models.load_model(model_arch)

    return model

def load_model(model, fname):
    model.load_weights(fname)

    return model

def main(loaded_model):
    cam = cv2.VideoCapture(0)
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

            age = np.squeeze(result[0]).round()
            gender = "Male" if np.squeeze(result[1]).round() == 0 else "Female"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0))
            cv2.putText(frame, f'Your {age} years old and {gender}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Press \'q\' to close..', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    if len(sys.argv) == 1:
        model_name = 'vgg16'
    else:
        if sys.argv[1] in ['vgg16', 'vgg19']:
            model_name = sys.argv[1]
        else:
            raise ValueError("Invalid Model Name, defined models are vgg{16, 19}")

    model_dir = './models/'

    model = create_model(model_dir+f'model_{model_name}.keras')
    loaded_model = load_model(model, model_dir+f'model_weights_{model_name}.hdf5')

    main(loaded_model)