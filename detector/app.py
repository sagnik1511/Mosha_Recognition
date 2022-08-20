import shutil
import numpy as np
from PIL import Image
from glob import glob
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File

MODEL = tf.keras.models.load_model("results/checkpoint")
classes = ["Aedes_aegypti", "Aedes_albopictus", "Culex_quinquefasciatus"]
app = FastAPI()

@app.post("/")
def upload_image(image: UploadFile = File(...)):

    # fetching file extension
    ext = image.filename.split('.')[-1]
    status = 202
    if ext in ["jpg", "png", "jpeg"]:
        # saving object in device memory
        with open(f"temp_image.{ext}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            buffer.close()
    else:
        status = 404
    return status

@app.get("/predict")
def predict():
    if len(glob("temp_image*")) > 0:
        # loading and processing image
        image = Image.open(glob("temp_image*")[0])
        image = image.resize((128, 128))
        image = np.expand_dims(np.array(image), 0)
        # predicting through model
        probs = MODEL.predict(image)
        class_id = np.argmax(probs, 1).reshape(-1).astype("int64")[0]
        class_name = classes[class_id]
        return {"class_name": class_name, "probability": probs[0][class_id] * 100.0}
    else:
        return {"No Image Found. Upload an Image to predict."}
    