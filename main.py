import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Set the MLflow tracking URI to your MLflow server
mlflow.set_tracking_uri("https://69fe-119-73-124-82.ngrok-free.app")

def main():
    # Load iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Set the experiment name
    mlflow.set_experiment("Iris_Classification")

    # Start an MLflow run
    with mlflow.start_run():
        # Set parameters for the model
        n_estimators = 100
        max_depth = 5

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
