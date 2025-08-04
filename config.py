# config.py
import os

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 2

DATA_DIR = "data/hvwc23"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
