
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from flask_cors import CORS  # Import CORS from flask_cors
import random
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load the saved model
model = tf.keras.models.load_model('image_classification_mod.h5')
img_width, img_height = 150, 150  # Same dimensions as during training
class_labels = ['Normal', 'Abnormal']

@app.route('/predict', methods=['POST'])
def predict_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'})

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected image'})
    
    

    # Save the uploaded image temporarily
    temp_image_path = 'temp_image.jpg'
    uploaded_file.save(temp_image_path)

    # Load the image and preprocess it for prediction
    img = image.load_img(temp_image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data

    prediction = model.predict(img_array)
    predicted_label = class_labels[np.argmax(prediction)]
    print(prediction,"prediction")


    response = {
        'name': predicted_label,   
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

