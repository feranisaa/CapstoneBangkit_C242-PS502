from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tempfile
import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import hashlib
from tensorflow.keras.layers import Layer


import tensorflow as tf
print(tf.__version__)


from firebase_admin import credentials, firestore
import firebase_admin

class L2Normalize(Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)

# Dapatkan referensi ke Firestore
db = firestore.client()

app = Flask(__name__)

# Konfigurasi URL model (GCS bucket publik)
MODEL_URL = os.getenv("MODEL_URL", "https://storage.googleapis.com/bucketsformodel/Recommender/recommender_fix_savemodel.h5")
IMAGE_MODEL_URL = os.getenv("IMAGE_MODEL_URL", "https://storage.googleapis.com/bucketsformodel/imageclass_model_new1.h5")
TEMP_DIR = tempfile.gettempdir()

# Fungsi untuk mengunduh model dari GCS
def download_model(model_url, model_path):
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully from {model_url}")
    else:
        raise Exception(f"Failed to download model from {model_url}, status code: {response.status_code}")

# Tentukan jalur model lokal
model_path = os.path.join(TEMP_DIR, 'recommender_fix_savemodel.h5')
image_model_path = os.path.join(TEMP_DIR, 'imageclass_model_new1.h5')

# Download model dari GCS (hanya sekali saat aplikasi dimulai)
download_model(MODEL_URL, model_path)
download_model(IMAGE_MODEL_URL, image_model_path)

# Memuat model
model = tf.keras.models.load_model(model_path, custom_objects={'l2_normalize': L2Normalize})
image_model = tf.keras.models.load_model(image_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request data
        input_data = request.json

        # Validate input data
        if not input_data or 'user_id' not in input_data:
            return jsonify({'error': 'Invalid input data'}), 400

        # Create a DataFrame from the input data
        num_rows = len(input_data['user_id'])
        data = pd.DataFrame({
            'user_id': input_data['user_id'],
            'rating_count': np.random.randint(5, 15, size=num_rows),
            'Zoo': np.random.uniform(3.0, 5.0, size=num_rows),
            'Historical & Museum': np.random.uniform(3.0, 5.0, size=num_rows),
            'Nature & Adventure & Park': np.random.uniform(3.0, 5.0, size=num_rows),
            'Waterpark': np.random.uniform(3.0, 5.0, size=num_rows),
            'Food': np.random.uniform(3.0, 5.0, size=num_rows),
        })

        # Prepare input features for the model
        input_features = data.drop(columns=['user_id']).values

        # Make predictions
        predictions = model.predict(input_features)

        # Process predictions
        relevance_scores = np.max(predictions, axis=1)  # Take the highest probability
        predicted_classes = np.argmax(predictions, axis=1)  # Get the index of the highest probability
        categories = ['Zoo', 'Historical & Museum', 'Nature & Adventure & Park', 'Waterpark', 'Food']
        predicted_categories = [categories[i] for i in predicted_classes]
        categories_data = pd.read_csv('DATA.csv')
        result = []
        for i, category in enumerate(predicted_categories):
            category_data = categories_data[categories_data['Kategori'] == category]
            result.append({
                'user_id': input_data['user_id'][i],
                'predicted_category': category,
                'places': category_data.to_dict(orient='records')
            })

         # Simpan hasil prediksi ke Firestore
        predictions_ref = db.collection('predictions')
        for record in result:
            predictions_ref.add(record)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/', methods=['POST'])
def image_predict():
    labels = ['Zoo', 'Historical & Museum', 'Nature & Adventure & Park', 'Waterpark', 'Food']
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imagefile.save(image_path)

    # Load gambar menggunakan Keras
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Prediksi kategori gambar
    yhat = image_model.predict(image)
    predicted_class = labels[yhat.argmax()]
    confidence = yhat.max()

    # Load data kategori dari CSV
    categories_data = pd.read_csv('DATA.csv')
    category_data = categories_data[categories_data['Kategori'] == predicted_class]

    # Simpan hasil prediksi ke Firestore
    image_predictions_ref = db.collection('image_predictions')
    image_predictions_ref.add({
        'prediction': predicted_class,
        'confidence': confidence * 100,
        'image_name': imagefile.filename,
        'places': category_data.to_dict(orient='records')
    })

    return jsonify({
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
        "places": category_data.to_dict(orient='records')
    })  

@app.route('/register', methods=['POST'])
def register_user():
    try:
        # Parse JSON request data
        user_data = request.json

        # Validate input data
        if not user_data or 'name' not in user_data or 'email' not in user_data or 'password' not in user_data:
            return jsonify({'error': 'Invalid input data'}), 400

        # Encrypt password (this example uses a simple hash; use a secure hashing algorithm like bcrypt for production)
        hashed_password = hashlib.sha256(user_data['password'].encode()).hexdigest()

        # Create a new user document
        user_ref = db.collection('users').add({
            'name': user_data['name'],
            'email': user_data['email'],
            'password': hashed_password  # Simpan password yang sudah dienkripsi
        })

        return jsonify({'message': 'User registered successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500



app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)

