from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
model = load_model('handwrittendigits.model')


@app.route('/')
def index():
    return render_template('index.html')

# Add a route for serving static files (e.g., PDF)
#to test
@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'), filename)


@app.route('/', methods=['POST'])
def predict_digit():
    try:
        # Get the uploaded file
        file = request.files['file']

        # Save the file
        filename = f"static/uploads/{file.filename}"
        file.save(filename)

        # Load the image
        img = cv2.imread(filename)[:, :, 0]

        # Invert the image
        img = np.invert(np.array([img]))

        # Predict the digit
        prediction = model.predict(img)

        # Get the predicted digit
        predicted_digit = np.argmax(prediction)

        return render_template('index.html', filename=filename, predicted_digit=predicted_digit)
    except Exception as e:
        print(e)
        return render_template('index.html', error="Error occurred during prediction.")


if __name__ == '__main__':
    app.run(debug=True)