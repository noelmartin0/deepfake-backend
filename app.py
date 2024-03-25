from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Construct the URL of the uploaded file
        file_url = request.host_url + 'uploads/' + filename
        return jsonify({'success': 'File uploaded successfully', 'file_url': file_url})

    return jsonify({'error': 'Error uploading file'})

@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    folder_path = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return jsonify({'success': 'All files deleted successfully'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/getVideo')
def get_video():
    # Fetch the list of uploaded files
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if uploaded_files:
        # Assuming you want to return the URL of the first uploaded file
        first_file = uploaded_files[0]
        file_url = request.host_url + 'uploads/' + first_file
        return jsonify({'file_url': file_url})
    else:
        return jsonify({'error': 'No uploaded files found'})

@app.route('/')
def home():
    return redirect(url_for('upload_file'))  # Redirect to the upload page

if __name__ == '__main__':
    app.run(debug=True)
