from .scripts.training_model import model
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
import requests
from io import BytesIO
import numpy as np


class ValidateImageService:
    """Validate image service."""

    def __init__(self, images):
        self.images_url = images
        self.good_images = []

    def validate_images(self):
        for url in self.images_url:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((700, 700))

            X = img_to_array(img)
            X = np.expand_dims(X, axis=0)
            images = np.vstack([X])

            try:
                prediction = model.predict(images)
                if prediction == 1:
                    self.good_images.append(url)
            except Exception as e:
                continue

        return self.good_images



