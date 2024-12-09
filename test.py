import json
import requests

payload = json.load(open("payload.json", "r"))

headers = {'Content-Type': 'application/json'}

url = "http://torchserve.local/predictions/coco-model"

response = requests.post(url, headers=headers, json=payload)

print(response.json())