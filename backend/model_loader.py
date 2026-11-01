import os
import urllib.request
from ultralytics import YOLO

# Resolve paths relative to this script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best.pt")

# Hugging Face direct link for OpenSistemas YOLOv8x Crack Segmentation weights
WEIGHTS_URL = "https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/resolve/main/yolov8x/weights/best.pt"

def get_crack_model():
    """
    Checks if local weights exist in backend/weights/best.pt.
    If missing, downloads them automatically from Hugging Face on server startup.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    if not os.path.exists(WEIGHTS_PATH):
        print("⏬ Downloading pre-trained YOLOv8 crack segmentation model...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("✅ Weights successfully downloaded to backend/weights/best.pt!")
    else:
        print("✅ Local weights loaded from backend/weights/best.pt")
        
    return YOLO(WEIGHTS_PATH)

if __name__ == "__main__":
    # Test model initialization
    model = get_crack_model()
    print("Model initialized successfully!")
