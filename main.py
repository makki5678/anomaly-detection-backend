from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

# Logger manually set up
logger = logging.getLogger("anomaly_detector")
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/upload/")
async def process_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Please upload a CSV file only.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception as e:
        logger.error(f"File reading failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to read CSV file.")

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        raise HTTPException(status_code=400, detail="No usable numeric data found in the uploaded CSV.")

    # Check if labels exist
    has_labels = 'label' in df_numeric.columns or 'Class' in df_numeric.columns

    # Separate features and labels
    if 'label' in df_numeric.columns:
        X = df_numeric.drop('label', axis=1)
        y_true = df_numeric['label']
    elif 'Class' in df_numeric.columns:
        X = df_numeric.drop('Class', axis=1)
        y_true = df_numeric['Class']
    else:
        X = df_numeric
        y_true = None

    # Train Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    preds = model.predict(X)

    preds = np.where(preds == -1, 1, 0)

    if y_true is not None:
        report = classification_report(y_true, preds, output_dict=True)
    else:
        report = None

    anomaly_rows = X[preds == 1]
    top_anomalies = anomaly_rows.head(5).to_dict(orient="records")

    return {
        "predictions": preds.tolist(),
        "anomaly_count": int(sum(preds)),
        "classification_report": report,
        "anomaly_explanations": top_anomalies,
    }
