import json
from locust import HttpUser, task
from PIL import Image
import base64
from io import BytesIO

url = "http://torchserve.local"

img_path = "/home/vdr/Projects/torchserve-helm/ml_app_2/african-wildlife-sample.jpg"

def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format=img.format)
        encoded_string = base64.b64encode(buffered.getvalue())
    return encoded_string

payload = json.load(open("payload.json", "r"))

headers = {'Content-Type': 'application/json'}

class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.post('/predictions/coco-model', json=payload, headers=headers)

    @task
    def hello_world2(self):
        self.client.post('/predictions/wildlife-model', json=payload, headers=headers)