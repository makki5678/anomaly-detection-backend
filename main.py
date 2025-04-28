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
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

app = FastAPI()

# Allow only frontend during dev
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

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    # Check if labels exist
    has_labels = False
    if df.columns[-1].strip().lower() == 'label':
        has_labels = True
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.copy()
        y = np.zeros(X.shape[0])

    # Keep only numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        raise HTTPException(status_code=422, detail="No numeric features found for modeling.")

    try:
        if has_labels:
            X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, stratify=y, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
    except Exception as e:
        logger.error(f"Data splitting issue: {str(e)}")
        raise HTTPException(status_code=500, detail="Problem splitting dataset.")

    try:
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(X_train)
    except Exception as e:
        logger.error(f"Model training issue: {str(e)}")
        raise HTTPException(status_code=500, detail="Model failed during training.")

    try:
        pred = clf.predict(X_test)
        pred = np.where(pred == 1, 0, 1)  # Flip: 0 = normal, 1 = anomaly

        results = {}

        if has_labels:
            report = classification_report(y_test, pred, output_dict=True)
            results['report'] = report
        else:
            results['report'] = None

        scores = clf.decision_function(X_test)
        top_anomalies = np.argsort(scores)[:5]

        important_features = []
        for idx in top_anomalies:
            row = X_test.iloc[idx]
            important = row.abs().sort_values(ascending=False).head(3)
            important_features.append(important.to_dict())

        results.update({
            "predictions": pred.tolist(),
            "num_anomalies": int(sum(pred)),
            "important_features": important_features
        })

        return results

    except Exception as e:
        logger.error(f"Prediction/reporting issue: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction phase failed.")
