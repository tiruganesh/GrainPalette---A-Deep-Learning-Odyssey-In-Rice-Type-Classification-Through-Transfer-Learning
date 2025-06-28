import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forms.forms import UploadForm

# Load model (ensure model.h5 is in the same directory as this script)
model = tf.keras.models.load_model("final_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/images'

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    prediction = None
    if form.validate_on_submit():
        f = form.image.data
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'Data', 'val')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, f.filename)
        f.save(filepath)

        a2 = cv2.imread(filepath)
        a2 = cv2.resize(a2, (128, 128))  # Use the correct size!
        a2 = np.array(a2) / 255.0
        a2 = np.expand_dims(a2, axis=0)

        pred = model.predict(a2)
        pred = pred.argmax()

        df_labels = {
            'arborio': 0,
            'basmati': 1,
            'ipsala': 2,
            'jasmine': 3,
            'karacadag': 4
        }

        prediction = "Unknown"
        for label, index in df_labels.items():
            if pred == index:
                prediction = label

        return render_template('results.html', prediction_text=prediction)
    return render_template('index.html', form=form)

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        f = request.files['image']
        filename = secure_filename(f.filename)
        static_img_dir = os.path.join(app.config['UPLOAD_FOLDER'])
        os.makedirs(static_img_dir, exist_ok=True)
        static_image_path = os.path.join(static_img_dir, filename)
        f.save(static_image_path)

        # Prediction logic
        a2 = cv2.imread(static_image_path)
        a2 = cv2.resize(a2, (128, 128))  # Change from (224, 224) to (128, 128)
        a2 = np.array(a2) / 255.0
        a2 = np.expand_dims(a2, axis=0)

        pred = model.predict(a2)
        pred = pred.argmax()

        df_labels = {
            'arborio': 0,
            'basmati': 1,
            'ipsala': 2,
            'jasmine': 3,
            'karacadag': 4
        }

        prediction = "Unknown"
        for label, index in df_labels.items():
            if pred == index:
                prediction = label

        image_path = f'images/{filename}'

        return render_template('results.html', prediction_text=prediction, image_path=image_path)
    return redirect('/')

@app.route('/details')
def pred():
    return render_template('details.html')

if __name__ == "__main__":
    app.run(debug=True)
