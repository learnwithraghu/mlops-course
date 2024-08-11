import bentoml
from bentoml.sklearn import save_model
import joblib

# Load the model from the .pkl file
model = joblib.load("model.pkl")

# Save the model into BentoML store
save_model("sentiment_analysis_model", model)
