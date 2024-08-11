import mlflow
import mlflow.sklearn
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download('movie_reviews')

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Separate the data into features and labels
texts = [" ".join(document) for document, category in documents]
labels = [category for document, category in documents]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a sentiment analysis pipeline
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model and log to MLflow
mlflow.set_experiment("Sentiment Analysis Experiment")
with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    
    # Save and log the model
    mlflow.sklearn.log_model(model, "sentiment_analysis_model")
    
    print(f"Model accuracy: {accuracy:.4f}")


