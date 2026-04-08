from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPrediction(BaseModel):
    prediction: str
    probability: list
