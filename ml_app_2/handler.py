import os
import base64
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

class WildlifeHandler(BaseHandler):
    def __init__(self):
        super(WildlifeHandler, self).__init__()

        self.imgsz = 640
        self.initialized = False
    
    def initialize(self, context):
        self.device = "cpu" # set to 0 for gpu

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        
        self.model = YOLO(self.model_pt_path, task="detection")
        self.model.to(self.device)
        
        self.initialized = True

    def decode_base64_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    
    def preprocess(self, data):
        images = []

        for row in data:
            input_data = row.get("data") or row.get("body")
            for image_b64 in input_data.get("images"):
                if isinstance(image_b64, str):
                    image = self.decode_base64_to_image(image_b64)
                    images.append(image)
        
        return images
    
    def inference(self, input_batch):
        predictions = self.model(input_batch, imgsz=self.imgsz)

        scores = []
        coordinates = []

        for pred in predictions:
            scores.append(pred.boxes.conf.cpu().numpy().tolist())
            coordinates.append(pred.boxes.xyxy.cpu().numpy().astype(np.int32).tolist())

        return {"scores": scores, "coordinates": coordinates}

    def postprocess(self, res):
        model_version = self.context.manifest['model']['modelVersion']

        result = {
            "model_version": model_version,
            "scores": res['scores'],
            "coordinates": res['coordinates']
        }
        
        return [result]