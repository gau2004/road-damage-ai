"""
app.py
Flask backend for the Road Damage Classification & Interactive
Retraining System.

Routes:
    GET  /            -> home page
    GET  /predict     -> prediction upload page
    POST /predict      -> run prediction on an uploaded image (AJAX, returns JSON)
    GET  /configure    -> configure & retrain page
    POST /retrain       -> retrain the model with given hyperparameters (AJAX, returns JSON)

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import os
import sys
import time
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))

from model.model_builder import build_model
from utils.preprocess import get_generators
from utils.metrics import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'road_model.keras')
CLASS_INFO_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ---- global in-memory state for the currently loaded model ----
current_model = None
current_image_size = 224
class_indices = {'Broken Road': 0, 'Not Broken Road': 1}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_current_model():
    """Load a previously trained model from disk, if one exists."""
    global current_model, current_image_size, class_indices
    if os.path.exists(MODEL_PATH):
        try:
            current_model = tf.keras.models.load_model(MODEL_PATH)
            current_image_size = current_model.input_shape[1]
        except Exception as e:
            print(f"Warning: could not load saved model ({e}).")
            current_model = None
    if os.path.exists(CLASS_INFO_PATH):
        with open(CLASS_INFO_PATH) as f:
            class_indices = json.load(f)


load_current_model()


@app.route('/')
def home():
    return render_template('index.html', model_ready=current_model is not None)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', model_ready=current_model is not None)

    if current_model is None:
        return jsonify({'error': 'No trained model found yet. Please train the model from the '
                                  '"Configure & Retrain" page first.'}), 400

    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use PNG, JPG, JPEG, BMP or WEBP.'}), 400

    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = keras_image.load_img(filepath, target_size=(current_image_size, current_image_size))
        arr = keras_image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = float(current_model.predict(arr, verbose=0)[0][0])
    except Exception as e:
        return jsonify({'error': f'Could not process image: {str(e)}'}), 400

    inv_class = {v: k for k, v in class_indices.items()}
    if pred > 0.5:
        label = inv_class.get(1, 'Class 1')
        confidence = pred * 100
    else:
        label = inv_class.get(0, 'Class 0')
        confidence = (1 - pred) * 100

    return jsonify({
        'prediction': label,
        'confidence': round(confidence, 2),
        'image_url': url_for('static', filename=f'uploads/{filename}')
    })


@app.route('/configure')
def configure():
    dataset_ready = _dataset_has_images()
    return render_template('retrain.html', dataset_ready=dataset_ready)


def _dataset_has_images():
    if not os.path.isdir(DATASET_DIR):
        return False
    for sub in os.listdir(DATASET_DIR):
        sub_path = os.path.join(DATASET_DIR, sub)
        if os.path.isdir(sub_path) and any(
            f.lower().endswith(tuple(ALLOWED_EXTENSIONS)) for f in os.listdir(sub_path)
        ):
            return True
    return False


@app.route('/retrain', methods=['POST'])
def retrain():
    global current_model, current_image_size, class_indices

    data = request.get_json(force=True)

    try:
        image_size = int(data.get('image_size', 224))
        batch_size = int(data.get('batch_size', 32))
        epochs = int(data.get('epochs', 10))
        learning_rate = float(data.get('learning_rate', 0.001))
        optimizer = data.get('optimizer', 'adam')
        activation = data.get('activation', 'relu')
        dropout = float(data.get('dropout', 0.3))
        augmentation = bool(data.get('augmentation', True))
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400

    if not _dataset_has_images():
        return jsonify({'error': 'Dataset not found. Place images inside '
                                  'dataset/Broken Road/ and dataset/Not Broken Road/ '
                                  'before retraining.'}), 400

    try:
        train_gen, val_gen = get_generators(DATASET_DIR, image_size, batch_size, augmentation)
    except Exception as e:
        return jsonify({'error': f'Error loading dataset: {str(e)}'}), 400

    if train_gen.samples == 0 or val_gen.samples == 0:
        return jsonify({'error': 'Not enough images found to create a train/validation split. '
                                  'Add more images to each class folder.'}), 400

    try:
        model = build_model(image_size, activation, dropout, learning_rate, optimizer)
        start = time.time()
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=2)
        training_time = time.time() - start
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

    model.save(MODEL_PATH)
    with open(CLASS_INFO_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f)

    acc_path, loss_path = plot_history(history, RESULTS_FOLDER)
    eval_metrics = evaluate_model(model, val_gen, RESULTS_FOLDER)

    # swap in the freshly trained model for prediction
    current_model = model
    current_image_size = image_size
    class_indices = train_gen.class_indices

    cache_bust = int(time.time())
    results = {
        'training_accuracy': round(float(history.history['accuracy'][-1]) * 100, 2),
        'validation_accuracy': round(float(history.history['val_accuracy'][-1]) * 100, 2),
        'training_loss': round(float(history.history['loss'][-1]), 4),
        'validation_loss': round(float(history.history['val_loss'][-1]), 4),
        'precision': round(eval_metrics['precision'] * 100, 2),
        'recall': round(eval_metrics['recall'] * 100, 2),
        'f1_score': round(eval_metrics['f1_score'] * 100, 2),
        'training_time_sec': round(training_time, 2),
        'confusion_matrix': eval_metrics['confusion_matrix'],
        'classes': list(train_gen.class_indices.keys()),
        'accuracy_graph': url_for('static', filename='results/accuracy.png') + f'?v={cache_bust}',
        'loss_graph': url_for('static', filename='results/loss.png') + f'?v={cache_bust}',
        'confusion_matrix_graph': url_for('static', filename='results/confusion_matrix.png') + f'?v={cache_bust}',
    }

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
