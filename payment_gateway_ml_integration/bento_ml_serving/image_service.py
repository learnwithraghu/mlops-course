import mlflow
import mlflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Normalize the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and log to MLflow
mlflow.set_experiment("Image Recognition Experiment")
with mlflow.start_run():
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
    
    # Save the model in the .h5 format for BentoML
    model.save("image_recognition_model.h5")
    
    # Log the model in MLflow
    mlflow.keras.log_model(model, "image_recognition_model.h5")
    
    print(f"Model accuracy: {accuracy:.4f}")
