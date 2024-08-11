import bentoml
from bentoml.io import JSON
import numpy as np

# Load the saved fraud detection model
model = bentoml.sklearn.load_model("fraud_detection_model")

# Create a BentoML service
svc = bentoml.Service("fraud_detection_service", runners=[])

# Define an API endpoint for fraud detection
@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    # Convert input data to a numpy array for model prediction
    data = np.array(input_data["features"]).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
