#!/usr/bin/python3

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def create_model():
    input_shape = (48, 48, 3)

    input_layer = tf.keras.layers.Input(shape=input_shape)
    pretrained = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=input_layer)

    flatten1 = tf.keras.layers.Flatten()(pretrained.output)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten1)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    dense1 = tf.keras.layers.Dense(1, name='age_output')(dense1)

    flatten2 = tf.keras.layers.Flatten()(pretrained.output)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(flatten2)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense2)
    dense2 = tf.keras.layers.Dense(1, activation='sigmoid', name='gender_output')(dense2)

    model = tf.keras.models.Model(inputs=pretrained.input, outputs=[dense1, dense2])

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0))
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0))
            cv2.putText(frame, str(result), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Press \'q\' to close..', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    model = create_model()
    loaded_model = load_model(model, 'model_weights.hdf5')
    main(loaded_model)