from pathlib import Path
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from core.detector import register_custom_modules
from ultralytics import YOLO
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_fixednames.pt"
app = FastAPI(title="Road Crack Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
model = None
@app.on_event("startup")
def load_model():
    global model
    register_custom_modules()
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
@app.get("/")
def home():
    return {"message": "Road Crack Detection API is running"}
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp.write(contents)
            temp_path = Path(temp.name)
        try:
            with Image.open(temp_path) as image:
                image.verify()
            results = model.predict(source=str(temp_path), verbose=False)
            result = results[0]
            detections = []
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls.item()),
                    "class_name": model.names[int(box.cls.item())],
                    "confidence": round(float(box.conf.item()), 4),
                    "box_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                })
            return {
                "filename": file.filename,
                "count": len(detections),
                "detections": detections,
            }
        finally:
            temp_path.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
    finally:
        await file.close()

