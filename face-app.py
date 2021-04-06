#!/usr/bin/python3

import flask
from flask import request, render_template

app = flask.Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run()