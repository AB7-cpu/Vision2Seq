"""
Config.py File to set and edit configurations for the model.
"""

from pathlib import Path

#    Directories:
ROOT_DIR       = Path(__file__).resolve().parent
DATA_DIR       = ROOT_DIR / "data" / "coco"
TRAIN_IMG_DIR  = DATA_DIR / "images" / "train2017"
VAL_IMG_DIR    = DATA_DIR / "images" / "val2017"
TRAIN_ANN      = DATA_DIR / "annotations" / "captions_train2017.json"
VAL_ANN        = DATA_DIR / "annotations" / "captions_val2017.json"
VOCAB_PATH     = ROOT_DIR / "data" / "vocabulary.pkl"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR    = ROOT_DIR / "results"

#    Vocabulary:
VOCAB_THRESHOLD = 5

#    Image preprocessing:
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

#    Model — V1 (CNN + LSTM):
EMBED_DIM  = 256  
HIDDEN_DIM = 512  
NUM_LAYERS = 1    
DROPOUT    = 0.5

#    Training:
BATCH_SIZE    = 64
NUM_EPOCHS    = 15
LEARNING_RATE = 4e-4
CNN_LR        = 1e-5
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 5.0
NUM_WORKERS   = 4
FINE_TUNE_CNN = True

# Training Settings:
SUBSET      = 0.4  # Percentage of the full 118k dataset.


SMOKE_TEST  = False # True: uses only the val2017 dataset just for initial testing.


RESUME_CHECKPOINT = "checkpoints/v1/best_model.pth"   # Uses best saved weight on retraining.

#    Inference:
BEAM_WIDTH  = 5
MAX_SEQ_LEN = 50

#    Logging:
LOG_INTERVAL  = 100
SAVE_INTERVAL = 1
USE_WANDB     = True
WANDB_PROJECT = "image-captioning-v1"
