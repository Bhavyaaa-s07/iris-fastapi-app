from fastapi import FastAPI
from .schemas import IrisInput, IrisPrediction
from .model import IrisModel

app = FastAPI(title="Iris Classification API")
model = IrisModel()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "iris_classifier_v1"}

@app.post("/predict", response_model=IrisPrediction)
def predict(payload: IrisInput):
    label, prob = model.predict(payload)
    return {"prediction": label, "probability": prob}