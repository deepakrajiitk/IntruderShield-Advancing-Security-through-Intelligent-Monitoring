from flask import Flask, render_template, request, redirect, url_for, Response
from flask import Flask, request, jsonify
from src.functions import create_directory, upload_photos
from src.recognizer_rtsp import Recognizer
from src.functions import update_ordered_dict
from collections import OrderedDict
import sys
import os
import cv2

# Add the path of the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)


def run(url, camera_id):
    cap = cv2.VideoCapture(url)
    rec_obj = Recognizer()
    # Initialize variables for face tracking
    prev_objects = OrderedDict()
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, known, unknown_count = rec_obj.recognizer(
                frame, prev_objects)
            prev_objects = update_ordered_dict(prev_objects, 0.2)
            rec_obj.save_logs(known, unknown_count)
            rec_obj.save_image(frame, camera_id+"_"+str(frame_count)+".jpg")
            # encode the frame as a JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_count += 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(run("test.mp4", "camera"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/create_directory', methods=['POST'])
def create_directory_route():
    username = request.form['username']
    if create_directory(username):
        return redirect(url_for('upload_photos_route', username=username))
    else:
        return render_template('directory_exists.html', username=username)


@app.route('/upload_photos/<username>', methods=['GET', 'POST'])
def upload_photos_route(username):
    if request.method == 'POST':
        # Get the files uploaded by the user
        files = request.files.getlist('photos')

        # Get the text input from the user
        text_input = request.form.get('text_input')

        return upload_photos(username, files, text_input)
    else:
        return render_template('upload.html', username=username)


if __name__ == '__main__':
    app.run(debug=True)
