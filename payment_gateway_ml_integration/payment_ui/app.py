from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def payment_form():
    return render_template('payment.html')

@app.route('/validate', methods=['POST'])
def validate_payment():
    # Extract form data
    card_number = request.form['card_number']
    expiration_date = request.form['expiration_date']
    cvv = request.form['cvv']
    amount = request.form['amount']
    billing_address = request.form['billing_address']
    zip_code = request.form['zip_code']
    email = request.form['email']

    # Create a feature vector from the form data
    # Here we're adding dummy features (e.g., 0, 1, or statistical defaults)
    features = [
        len(card_number), float(amount), len(billing_address), int(cvv), len(zip_code), len(email),
        0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1  # Dummy or default values for additional features
    ]

    # Call the BentoML fraud detection service
    response = requests.post("http://127.0.0.1:3000/predict", json={"features": features})
    prediction = response.json()["prediction"]

    # Check if the transaction is fraudulent
    if prediction == 1:
        return "Transaction Blocked: Potential Fraud Detected!"
    else:
        return "Payment Successful!"

if __name__ == '__main__':
    app.run(debug=True, port=5003)
