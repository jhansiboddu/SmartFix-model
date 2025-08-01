import os
import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class_names = ["electrical", "plumbing", "civil"]
text_inputs = clip.tokenize([f"a photo of {label} damage" for label in class_names]).to(device)

def predict_image_class(image_path: str):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        predicted = class_names[probs.argmax()]
        confidence = float(probs.max())
        return {
            "predicted_class": predicted,
            "confidence": f"{confidence:.2f}"
        }
    except Exception as e:
        return {"error": str(e)}
