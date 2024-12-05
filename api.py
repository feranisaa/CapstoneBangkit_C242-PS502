from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import os
import tempfile
import requests

app = Flask(__name__)

# Konfigurasi URL model
MODEL_URL = os.getenv("MODEL_URL", "https://storage.googleapis.com/my-bucket/imageclass_model.h5")

# Download dan cache model
response = requests.get(MODEL_URL)
if response.status_code != 200:
    raise ValueError(f"Failed to download model. HTTP status code: {response.status_code}")

with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
    tmp_file.write(response.content)
    model_path = tmp_file.name
    model = load_model(model_path)

# Baca label dari file
labels = ['historical', 'makanan', 'museum', 'nature_adventure', 'park', 'waterpark', 'zoo']

# Batasan ukuran upload file
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Maksimal 2 MB

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename

    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imagefile.save(image_path)

    # Preproses gambar
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Prediksi dengan model
    yhat = model.predict(image)
    predicted_class = labels[yhat.argmax()]
    confidence = yhat.max()

    # Kembalikan respons JSON
    return jsonify({
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
