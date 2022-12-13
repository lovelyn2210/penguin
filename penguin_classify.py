from flask import jsonify
from tensorflow import keras
from PIL import Image
import io
import tensorflow as tf
import numpy as np


imageClassificationModels = keras.models.load_model("resnet_model.h5")
labels = ["0", "1"]


def classification(image_data):
    imageFile = Image.open(io.BytesIO(image_data))
    # imageFile = imageFile.convert('RGB')
    results = {};
    for i in range(6):
        imgI = imageFile.crop(((i%3)*98+2,(i//3)*98+2,(i%3)*98+2+96,(i//3)*98+2+96))
        imgI = keras.preprocessing.image.img_to_array(imgI)
        img_array = tf.expand_dims(imgI, 0)  # Create batch axis
        predictions = imageClassificationModels.predict(img_array)
        label = labels[ np.argmax(predictions[0])]
        results[i+1] = label
    return jsonify(results)