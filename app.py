from flask import Flask, render_template, request, redirect, url_for
from src.functions import create_directory, upload_photos

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

if __name__ == '__main__':
    app.run(debug=True)
