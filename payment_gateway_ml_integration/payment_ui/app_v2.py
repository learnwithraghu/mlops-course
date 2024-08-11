from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def payment_form():
    return render_template('payment.html')

@app.route('/upload_image')
def upload_image():
    return render_template('upload.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Extract image from request
    file = request.files['file']
    image = Image.open(file.stream)

    # Convert image to bytes for BentoML API call
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Call BentoML service with the image data
    response = requests.post("http://127.0.0.1:3000/predict_image", files={"file": img_byte_arr})
    data = response.json()

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=5003)
