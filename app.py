from flask import Flask, render_template, request, redirect, url_for
from flask import Flask, request, jsonify
from src.functions import create_directory, upload_photos
import sys
import os
import argparse
# Add the path of the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)

@app.route('/')
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
    

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    from src.recognizer import recognizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='dataset/test/ch24', help='path to the input directory containing the images to be recognized')
    ap.add_argument('--output-dir', default='results', help='path to the output directory where the result images will be saved')
    ap.add_argument('--embed-dir', default='dataset/cropped_authorized', help='path to the directory containing the authorized embeddings')
    ap.add_argument('--rec-model-name', default='w600k_r50.onnx', help='recognizer model name')
    ap.add_argument('--detector-thres', default=0.8, help='confidence threshold for face detection')

    args = ap.parse_args()

    try:
        # recognizer(args)
        # read the content of the file
        with open('results/run5/logs.txt', 'r') as f:
            content = f.read()

        # split the content into a list of lines
        lines = content.split('\n')

        # render the template with the lines as a context variable
        return render_template('show_logs.html', lines=lines)
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)