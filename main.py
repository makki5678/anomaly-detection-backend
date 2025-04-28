from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import io
import logging

# Configure logging
logger = logging.getLogger("anomaly_detector")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="Anomaly Detection Service", version="1.0.0")

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "Anomaly Detection API is operational."}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Only CSV is allowed.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except UnicodeDecodeError as e:
        logger.error(f"Decoding error: {e}")
        raise HTTPException(status_code=400, detail="Could not decode uploaded file.")
    except Exception as e:
        logger.error(f"Failed to process CSV: {e}")
        raise HTTPException(status_code=400, detail="Error processing CSV file.")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    y_true = None
    if "Class" in df.columns:
        y_true = df.pop("Class")

    try:
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(df)
        raw_preds = model.predict(df)
        preds = np.where(raw_preds == -1, 1, 0)
    except Exception as e:
        logger.error(f"Model training or prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Anomaly detection failed.")

    classification = classification_report(y_true, preds, output_dict=True) if y_true is not None else None

    anomalies_detected = df[preds == 1]
    example_anomalies = anomalies_detected.head(5).to_dict(orient="records") if not anomalies_detected.empty else []

    return {
        "predictions": preds.tolist(),
        "anomaly_count": int(preds.sum()),
        "classification_report": classification,
        "sample_anomalies": example_anomalies
    }
