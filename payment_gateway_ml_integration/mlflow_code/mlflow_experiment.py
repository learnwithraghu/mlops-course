import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Generate synthetic data for fraud detection
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(probability=True)
}

# Start an MLflow experiment
mlflow.set_experiment("Fraud Detection Experiment")

best_model_name = None
best_model_uri = None
best_accuracy = 0

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log metrics in MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model in MLflow
        model_info = mlflow.sklearn.log_model(model, model_name)

        print(f"Model: {model_name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Check if this model is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model_uri = model_info.model_uri

print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Register the best model in the MLflow Model Registry
if best_model_uri:
    mlflow.register_model(model_uri=best_model_uri, name="FraudDetectionModel")
    print(f"Model registered in the Model Registry as 'FraudDetectionModel'.")
