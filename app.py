import os
import cv2
import numpy as np
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "uploads")
FRAME_FOLDER = os.path.join(os.path.dirname(__file__), "static", "frames")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["FRAME_FOLDER"] = FRAME_FOLDER


@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=1)


model_path = (
    r"D:\DeepFake2.0\project\deepfake_efficientnetb2_transformer_balanced_focal2.keras"
)
model = load_model(model_path, custom_objects={"focal_loss": focal_loss})

feature_extractor = EfficientNetB2(
    include_top=False, pooling="avg", input_shape=(224, 224, 3)
)


def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    return efficientnet_preprocess(frame_resized)


def extract_frames_features(video_path, desired_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return None

    indices = (
        np.linspace(0, frame_count - 1, desired_frames).astype(int)
        if frame_count >= desired_frames
        else np.arange(frame_count)
    )
    frames_features = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = preprocess_frame(frame_rgb)
        feature_vector = feature_extractor.predict(np.expand_dims(processed, axis=0))
        frames_features.append(feature_vector[0])

    cap.release()
    frames_features = np.array(frames_features)

    if frames_features.shape[0] < desired_frames:
        pad_len = desired_frames - frames_features.shape[0]
        padding = np.zeros((pad_len, 1408))
        frames_features = np.vstack([frames_features, padding])
    elif frames_features.shape[0] > desired_frames:
        frames_features = frames_features[:desired_frames]

    return np.expand_dims(frames_features, axis=0)


def save_middle_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return False

    middle_frame_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
    cap.release()
    return ret


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files or request.files["video"].filename == "":
        return redirect(url_for("index"))

    file = request.files["video"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract features for prediction
    features = extract_frames_features(filepath)
    if features is None:
        return render_template(
            "result.html", error="Could not extract frames from video"
        )

    preds = model.predict(features)
    confidence = float(preds[0][0])
    result = "Fake" if confidence > 0.77 else "Real"
    confidence_pct = round(confidence * 100, 2)
    file_size = round(os.path.getsize(filepath) / (1024 * 1024), 2)  # MB

    # Save a single middle frame for preview
    frame_filename = f"{os.path.splitext(filename)[0]}_frame.jpg"
    frame_filepath = os.path.join(app.config["FRAME_FOLDER"], frame_filename)
    saved = save_middle_frame(filepath, frame_filepath)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence_pct,
        filename=filename,
        size=file_size,
        frame_filename=frame_filename if saved else None,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/static/frames/<filename>")
def frame_file(filename):
    return send_from_directory(app.config["FRAME_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
