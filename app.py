# app.py
from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pickle
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load precomputed embeddings
with open("image_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Similarity function
def get_top_k_similar(uploaded_image_path, embeddings, k=20):
    image = Image.open(uploaded_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        query_emb = model.get_image_features(**inputs)
    query_emb = query_emb / query_emb.norm()
    
    similarities = {}
    for img_name, emb in embeddings.items():
        sim = torch.cosine_similarity(query_emb, emb)
        similarities[img_name] = sim.item()
    
    top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    return [img for img, score in top_k]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    top_images = get_top_k_similar(filepath, embeddings)
    return jsonify({"top_images": top_images})

if __name__ == "__main__":
    app.run(debug=True)
