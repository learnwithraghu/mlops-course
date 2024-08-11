import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import bentoml

# Generate a synthetic dataset for fraud detection
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model with BentoML
print("Saving the model")
# Save the model with BentoML and get the model save information
model_info = bentoml.sklearn.save_model("fraud_detection_model", model)

# Print the path where the model is saved
print(f"Model saved at: {model_info.path}")

print("Model saved")
