from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

app = FastAPI()

IMG_SIZE = 112

emotion_labels = [
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

model = tf.saved_model.load("models/emotion_savedmodel")
print("Model loaded")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def preprocess(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # convert to grayscale (model expects 1 channel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray / 255.0

    gray = np.expand_dims(gray, axis=-1)  # channel
    gray = np.expand_dims(gray, axis=0)   # batch

    return gray


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess(image_bytes)

        preds = model.predict(img, verbose=0)

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return {
            "emotion": emotion_labels[idx],
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
