#!/usr/bin/python3

import os
import sys
import cv2
import tensorflow as tf

def create_model():
    input_shape = (48, 48, 3)

    input_layer = tf.keras.layers.Input(shape=input_shape)
    pretrained = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=input_layer)

    flatten = tf.keras.layers.Flatten()(pretrained.output)
    dense = tf.keras.layers.Dense(128, activation='relu')(flatten)
    dense = tf.keras.layers.Dense(64, activation='relu')(dense)
    dense = tf.keras.layers.Dense(2)(dense)

    model = tf.keras.models.Model(input=pretrained.input, output=dense)

    return model

def load_model(model, fname):
    model.load_model(fname)

    return model

def main(model):
    cam = cv2.VideoCapture(0)
    clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if not cam.isOpened():
        print("Cannot open camera exiting......")
        sys.exit()

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        faces = clf.detectMultiScale(frame)

        for face in faces:
            (x, y, w, h) = [v for v in face]

        img_resized = cv2.resize(frame)

        cv2.imshow('Main Window', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()