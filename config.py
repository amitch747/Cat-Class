from pathlib import Path

# Main paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Best Models
PYTORCH_MODEL_PATH = MODELS_DIR / "best.pt"
ONNX_MODEL_PATH = MODELS_DIR / "best.onnx"
ENGINE_MODEL_PATH = MODELS_DIR / "best.engine"

CLASS_LIST = ['Finn', 'Rue', 'Yeager']
CLASS_NAMES = {
    0: 'Finn',
    1: 'Rue',
    2: 'Yeager'
}
CLASS_COLORS = {
    0: (100, 255, 100),  # Light green  Finn
    1: (100, 100, 255),  # Light red  Rue
    2: (255, 100, 100),  # Light blue Yeager
}

# Additional Parameters
IMG_SIZE = 640 
NUM_CLASSES = len(CLASS_NAMES)
CONFIDENCE_THRESHOLD = 0.7
NMS_IOU_THRESHOLD = 0.4

INPUT_NAME = "images"
OUTPUT_NAME = "output0"
