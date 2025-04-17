from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Load the trained model
model = tf.keras.models.load_model('model/deepfake_video_detector_2.keras')

# Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract exactly 32 frames, padding if needed
def extract_frames(video_path, num_frames=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    interval = max(1, total_frames // num_frames)
    count = 0

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (300, 300))
            frame = frame.astype('float32') / 255.0  # Normalize
            frames.append(frame)
        count += 1

    cap.release()

    # Pad with last frame if needed
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])

    return np.array(frames)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded!'})

    video = request.files['video']

    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Clear previous uploads
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        video.save(upload_path)

        frames = extract_frames(upload_path)

        if len(frames) == 0:
            return jsonify({'error': 'Could not process video.'})

        # Predict each frame individually
        frame_preds = []
        for frame in frames:
            frame_input = np.expand_dims(frame, axis=0)  # Shape: (1, 300, 300, 3)
            pred = model.predict(frame_input, verbose=0)[0][0]  # Single value
            frame_preds.append(pred)

        prediction = np.mean(frame_preds)

        result = 'Real' if prediction >= 0.5*100 else 'Fake'
        confidence = round(prediction * 100 if result == 'Real' else (100 - prediction * 100), 2)

        return jsonify({
            'result': result,
            'confidence': confidence,
            'video_path': upload_path
        })

    return jsonify({'error': 'Invalid file type.'})

if __name__ == '__main__':
    app.run(debug=True)
