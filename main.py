import os
import threading
import time
from io import BytesIO

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = "faceDetectionModel.pth"


def read_image_from_url(url: str):
    response = requests.get(url, timeout=5)
    if response.status_code != 200:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def preprocess_image(image):
    transform = transforms.Compose(
        [transforms.Resize((160, 160)), transforms.ToTensor()]
    )
    return (
        transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    )


class SimpleFaceNet(nn.Module):
    def __init__(self):
        super(SimpleFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        dummy_input = torch.zeros(1, 3, 160, 160)
        with torch.no_grad():
            dummy_output = self._forward_conv_layers(dummy_input)
        flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)

    def _forward_conv_layers(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


facenet = SimpleFaceNet().to("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(MODEL_PATH):
    facenet.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
else:
    torch.save(facenet.state_dict(), MODEL_PATH)


@torch.no_grad()
def get_face_embedding(image):
    image_tensor = preprocess_image(image)
    return facenet(image_tensor)


@torch.no_grad()
def compare_faces(face1, face2):
    return F.cosine_similarity(face1, face2).item()


class ImageURLs(BaseModel):
    image1_url: str
    image2_url: str


def process_images(image_urls, result):
    img1 = read_image_from_url(image_urls.image1_url)
    img2 = read_image_from_url(image_urls.image2_url)

    if img1 is None or img2 is None:
        result["error"] = "Invalid image URL or unable to fetch image"
        return

    embedding1 = get_face_embedding(img1)
    embedding2 = get_face_embedding(img2)

    result["similarity"] = compare_faces(embedding1, embedding2)


@app.post("/compare_faces/")
async def compare_faces_api(image_urls: ImageURLs):
    result = {}
    thread = threading.Thread(target=process_images, args=(image_urls, result))
    thread.start()
    thread.join()

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"similarity": result["similarity"]}
