import bentoml
from bentoml.io import JSON
import numpy as np

# Load the saved model from BentoML's model store
model = bentoml.sklearn.load_model("sentiment_analysis_model:latest")

# Create a BentoML service
svc = bentoml.Service("sentiment_analysis_service", runners=[])

# Define an API endpoint for sentiment analysis
@svc.api(input=JSON(), output=JSON())
def predict_sentiment(input_data: dict):
    text = input_data["text"]
    prediction = model.predict([text])[0]

    sentiment_map = {'1': 'positive', '0': 'negative'}
    sentiment = sentiment_map.get(str(prediction), 'neutral')
    
    return {"sentiment": sentiment}
