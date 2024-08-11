import bentoml
from bentoml.io import JSON
import numpy as np

# Load the saved Random Forest model from BentoML's model store
model = bentoml.sklearn.load_model("fraud_detection_model:latest")

# Create a BentoML service
svc = bentoml.Service("fraud_detection_service", runners=[])

# Define an API endpoint for fraud detection
@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    # Extract features from the input data
    features = np.array(input_data["features"]).reshape(1, -1)
    
    # Get the prediction from the model
    prediction = model.predict(features)
    
    # Return the prediction as JSON
    return {"prediction": int(prediction[0])}
