# embeddings.py
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_folder = "images/"
embeddings = {}

# Precompute embeddings
for img_file in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_file)
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    embeddings[img_file] = emb / emb.norm()  # normalize

# Save embeddings to disk
with open("image_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings saved!")
