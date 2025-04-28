from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import io
import logging

# Initialize logger
logger = logging.getLogger("anomaly_detector")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="Anomaly Detection API", version="1.0")

# CORS setup (specific origins should be preferred in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Anomaly Detection API is live."}

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a CSV.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except UnicodeDecodeError as e:
        logger.error(f"File decoding error: {e}")
        raise HTTPException(status_code=400, detail="Unable to decode uploaded file.")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise HTTPException(status_code=400, detail="Failed to process CSV.")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    y_true = None
    if "label" in df.columns:
        y_true = df.pop("label")

    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(df)
        pred_raw = model.predict(df)
        preds = np.where(pred_raw == -1, 1, 0)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail="Model training failed.")

    report = classification_report(y_true, preds, output_dict=True) if y_true is not None else None

    anomalies = df[preds == 1]
    sample_anomalies = anomalies.head(5).to_dict(orient="records") if not anomalies.empty else []

    response = {
        "anomaly_count": int(np.sum(preds)),
        "predictions": preds.tolist(),
        "classification_report": report,
        "anomaly_samples": sample_anomalies
    }

    return response
