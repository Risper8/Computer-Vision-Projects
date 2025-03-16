from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model('gender_model.keras')

def process_image(image_path, target_size=(56, 56)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image 


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    base_path = 'sample images'
    filename = 'female(2).jpg'
    image_path = os.path.join(base_path, filename)
    file.save(image_path)
    image = process_image(image_path)
    prediction = model.predict(image)
    if prediction > 0.5:
        prediction_class=1
    else:
        prediction_class=0
    # Interpret the results
    class_labels = ['female', 'male']
    return jsonify(f'Predicted class: {class_labels[prediction_class]}')

if __name__ == '__main__':
    app.run(debug=True)
