from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# 1) FastAPI app & static mounting
app = FastAPI(title="Diabetes Classifier")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) Serve index.html at root
@app.get("/", response_class=FileResponse)
def read_index():
    return os.path.join("static", "index.html")

# 3) Request schema (8 raw Pima features)
class DiabetesRequest(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

# 4) Model definition matching Colab
class PimaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# 5) Load artifacts
BASE = os.path.dirname(__file__)
model = PimaNet()
model.load_state_dict(torch.load(os.path.join(BASE, "pima_model.pth"), map_location="cpu"))
model.eval()

scaler_X = joblib.load(os.path.join(BASE, "pima_scaler_X.pkl"))

# 6) Prediction endpoint
@app.post("/predict")
def predict(req: DiabetesRequest):
    # raw input array
    raw = np.array([[ 
        req.pregnancies,
        req.glucose,
        req.blood_pressure,
        req.skin_thickness,
        req.insulin,
        req.bmi,
        req.diabetes_pedigree,
        req.age
    ]], dtype=float)
    # scale, infer, sigmoid→prob
    Xs = scaler_X.transform(raw)
    with torch.no_grad():
        logits = model(torch.tensor(Xs, dtype=torch.float32))
        prob = torch.sigmoid(logits).item()
    pred = "diabetic" if prob > 0.5 else "not diabetic"
    return {"prediction": pred, "probability": prob}

# 7) Health‐check
@app.get("/health")
def health():
    return {"status": "ok"}

# 8) Run
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
