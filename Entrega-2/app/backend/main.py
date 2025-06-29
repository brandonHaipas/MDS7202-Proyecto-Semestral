import os
import mlflow
import uvicorn
import pandas as pd
from typing import Dict
from fastapi import FastAPI, Body
from mlflow.tracking import MlflowClient

# Carga de modelo
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")
MODEL_NAME   = "best_model" 
MODEL_ALIAS  = "current"
client = MlflowClient()
mlflow.set_tracking_uri("file:/mlruns")

mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

# crear aplicación
app = FastAPI()

@app.post("/predict")
async def predict(features: Dict = Body(...)):
    prediction = model.predict(
        pd.DataFrame(
            [features]
        )
    ).tolist()[0]

    return {"prediction": prediction}

if __name__ == '__main__':
    uvicorn.run('main:app', port = 8000)