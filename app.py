from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from predict import prediction

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
result = ""

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'flv', 'wmv'}

@app.route('/upload', methods=['POST'])
def upload_file():
    global result
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Call prediction function from predict.py
        result = prediction(file_path)
        print("The video is: ",result)
        return jsonify({'success': 'File uploaded successfully', 'prediction_result': result})

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
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    print("Data is: ",uploaded_files)
    if uploaded_files:
        first_file = uploaded_files[0]
        file_url = request.host_url + 'uploads/' + first_file
        return jsonify({'file_url': file_url,'result':result})
    else:
        return jsonify({'error': 'No uploaded files found'})
    

@app.route('/')
def home():
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
