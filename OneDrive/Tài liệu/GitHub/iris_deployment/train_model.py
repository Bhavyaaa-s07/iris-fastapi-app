# If you don't have the .pkl file yet, run this script to generate it
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

def train():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train a simple model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    joblib.dump(clf, 'models/iris_model.pkl')
    print("Model saved to models/iris_model.pkl")

if __name__ == "__main__":
    train()