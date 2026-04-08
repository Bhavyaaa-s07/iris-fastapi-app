import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "iris_model.pkl")

class IrisModel:
    def __init__(self):
        # Load the model during initialization
        self.model = joblib.load(MODEL_PATH)
        self.target_names = ['setosa', 'versicolor', 'virginica']

    def predict(self, data):
        features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        prediction = self.model.predict(features)
        probability = self.model.predict_proba(features).tolist()[0]
        return self.target_names[prediction[0]], probability
