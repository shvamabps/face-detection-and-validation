import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI(docs_url=None, redoc_url=None)


class ImageURLs(BaseModel):
    image1_url: str
    image2_url: str


def read_image_from_url(url: str):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return None
    return Image.open(response.raw).convert("RGB")


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
        self.fc1 = nn.Linear(128 * 20 * 20, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


facenet = SimpleFaceNet().to("cuda" if torch.cuda.is_available() else "cpu")


def get_face_embedding(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        embedding = facenet(image_tensor)
    return embedding


def compare_faces(face1, face2):
    return F.cosine_similarity(face1, face2).item()


@app.post("/compare_faces/")
async def compare_faces_api(image_urls: ImageURLs):
    img1 = read_image_from_url(image_urls.image1_url)
    img2 = read_image_from_url(image_urls.image2_url)

    if img1 is None or img2 is None:
        raise HTTPException(
            status_code=400, detail="Invalid image URL or unable to fetch image"
        )

    embedding1 = get_face_embedding(img1)
    embedding2 = get_face_embedding(img2)

    similarity = compare_faces(embedding1, embedding2)

    return {"similarity_score": similarity}
