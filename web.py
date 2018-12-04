from flask import Flask, render_template, Response, request
from PIL import Image, ImageDraw
from time import time, sleep
import io
import numpy as np
import threading
app = Flask(__name__)

frame = None
@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')

def gen():
    while True:
        global frame
        if frame != None:
            yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/put_image", methods=['POST'])
def put_image():
    global frame
    frame = request.data
    return Response(status=200)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')

