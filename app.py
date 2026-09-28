from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import io
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

app = FastAPI(title="Smart Trash Alert ML Prototype")

# Download the YOLOv8 garbage classification model
print("Loading YOLOv8 model from Hugging Face...")
model_path = hf_hub_download(repo_id="kendrickfff/YOLOv8GarbageClassification", filename="best v2.pt")
model = YOLO(model_path)
print("Model loaded successfully.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Run YOLOv8 prediction
    results = model(image)
    
    names = model.names
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the first bounding box (or highest confidence)
        boxes = results[0].boxes
        # Get the box with the max confidence
        best_box = max(boxes, key=lambda x: x.conf[0].item())
        class_id = int(best_box.cls[0].item())
        conf = best_box.conf[0].item()
        
        # Translate to Indonesian (optional, but good for UI)
        translation = {
            "battery": "Baterai",
            "biological": "Organik / Biologis",
            "brown-glass": "Kaca Cokelat",
            "cardboard": "Kardus",
            "clothes": "Pakaian",
            "green-glass": "Kaca Hijau",
            "metal": "Logam / Kaleng",
            "paper": "Kertas",
            "plastic": "Plastik",
            "shoes": "Sepatu",
            "trash": "Sampah Campuran",
            "white-glass": "Kaca Bening"
        }
        
        class_name = names[class_id]
        translated_name = translation.get(class_name, class_name)
        
        return {"class_name": translated_name, "confidence": conf, "original_class": class_name}
    else:
        return {"class_name": "Tidak Terdeteksi", "confidence": 0.0, "original_class": "None"}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("Smart Trash Alert.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
