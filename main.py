import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, Response


app = Flask(__name__)
# camera = cv2.VideoCapture(0)

if os.environ.get("WERKZEUG_RUN_MAIN") or Flask.debug is False:
    camera = cv2.VideoCapture(0)


labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model('model-090.model')

def genFrames():
    while True:
        success, frame = camera.read()
        frame = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_rects:
            face_img = gray[y+(abs(w - h)):y+h, x:x+w]
            resized = cv2.add(cv2.resize(face_img, (100, 100)), 60)
            normalized = resized/255.0

            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 4)
            cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], 4)
            cv2.putText(frame, labels_dict[label], (x, y-10),
                        cv2.FONT_ITALIC, 1, (255, 255, 255), 4) 


            break

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/service")
def service():
    return render_template("service.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/trynow")
def trynow():
    return render_template("TryNow.html")


def gen(camera):
    print("gen")
    frame = camera.get_frame()
    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(genFrames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
