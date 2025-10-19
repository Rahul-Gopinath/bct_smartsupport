from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from error_detector import detect_errors
from suggestions import get_suggestion

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---- Routes ----

@app.route('/')
def home():
    # Renders templates/index.html
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    uploaded_files = request.files.getlist('files')
    file_paths = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_paths.append(file_path)

    lines = detect_errors(file_paths)
    return jsonify({'highlightedLines': lines})

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    if not data or 'lineText' not in data:
        return jsonify({'error': 'Missing line text'}), 400

    line_text = data['lineText']
    suggestion = get_suggestion(line_text)
    return jsonify({'suggestion': suggestion})


if __name__ == '__main__':
    app.run(debug=True)
