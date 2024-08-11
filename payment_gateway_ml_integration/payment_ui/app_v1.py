from flask import Flask, render_template, request, redirect, url_for

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

    # Here you would call your BentoML service and pass the form data
    # For now, we'll just simulate the logic:
    
    # Simulate a fraud check (this would be replaced by your BentoML API call)
    is_fraudulent = False  # Replace this with actual prediction logic
    
    if is_fraudulent:
        return "Transaction Blocked: Potential Fraud Detected!"
    else:
        return "Payment Successful!"

if __name__ == '__main__':
    app.run(debug=True, port=5003)
