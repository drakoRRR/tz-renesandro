from .scripts.training_model import model
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
import requests
import openai
from io import BytesIO
import numpy as np


from application.settings import OPENAI_KEY


class ValidateImageService:
    """Validate image service."""

    def __init__(self, images):
        self.images_url = images
        self.good_images = []
        openai.api_key = OPENAI_KEY
        self.client = openai.OpenAI()
        self.bugs = ("more or less than five fingers on one arm, more or less than 2 arms or lags, two heads, "
                     "disproportionately long or short fingers, disproportionately long or short limbs, weird pose or "
                     "body plastic of people, parts of the body are separate from the body, limbs of some people stuck "
                     "in others, disproportionate head size, crooked limbs, crooked fingers, blurred face, "
                     "flying objects")

    def validate_images(self, enable_gpt=False):
        """Validate images to determine if they have been properly generated using our own model."""
        for url in self.images_url:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((700, 700))

            X = img_to_array(img)
            X = np.expand_dims(X, axis=0)
            images = np.vstack([X])

            try:
                model_prediction = model.predict(images)
                if enable_gpt:
                    gpt_prediction = self.validate_with_chatgpt(url)
                    model_prediction = self._get_prediction(model_prediction, gpt_prediction)

                if model_prediction == 1:
                    self.good_images.append(url)
            except Exception as e:
                continue

        return self.good_images

    def validate_with_chatgpt(self, image_url):
        """Validate image through openai api."""

        payload = {"model": "gpt-4-vision-preview",
                   "messages": [
                       {"role": "system",
                        "content": [{"type": "text",
                                     "text": f"You are an experienced image analyst tasked with identifying defects "
                                             f"in various types of images, including those related to anime, cartoons, "
                                             f"and graphic images, as well as realistic depictions of people. Your goal "
                                             f"is to thoroughly examine each image for any abnormalities or defects, "
                                             f"such as {self.bugs}. While analyzing images, pay attention to both "
                                             f"common defects and unique characteristics of different genres, "
                                             f"including unconventional proportions, exaggerated features, "
                                             f"and stylized artwork often found in anime. Additionally, ensure accurate"
                                             f" identification of human subjects and their body parts. Your analysis"
                                             f" should be comprehensive and detailed to ensure accurate defect"
                                             f" detection across different image types."
                                        }],
                        },
                       {
                           "role": "user",
                           "content": [
                               {
                                   "type": "text",
                                   "text": "If image is invalid, answer just 0, if picture is valid, write 1"
                               },
                               {
                                   "type": "image_url",
                                   "image_url": {
                                       "url": image_url
                                   }
                               }
                           ]
                       }
                   ],
                   "max_tokens": 500
                   }

        headers = {"Authorization": f"Bearer {OPENAI_KEY}",
                   "Content-Type": "application/json"}

        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
        r = response.json()
        return int(r["choices"][0]["message"]["content"])

    @staticmethod
    def _get_prediction(model_prediction, gpt_prediction):
        """Result of both predictions."""
        if model_prediction == gpt_prediction:
            return model_prediction
        return gpt_prediction

