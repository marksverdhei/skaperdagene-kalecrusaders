from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("/mnt/c/Users/a5050rs/Downloads/andreapose.jpg")
# image = Image.open("/mnt/c/Users/a5050rs/Downloads/andreabadpose.jpg")
# image = Image.open("/mnt/c/Users/a5050rs/Downloads/nerd-neck-forward-head-posture-fix-jpg.jpg")


inputs = processor(text=["good posture, straight neck, shoulders relaxed", "bad posture, neck bent, leaning"], images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)