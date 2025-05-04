import torch

class Config:
    # Data parameters
    SAMPLE_RATE = 16000
    DURATION = 4  # seconds
    N_FFT = 1024
    HOP_LENGTH = 64
    WIN_LENGTH = 512
    N_MELS = 128
    SPECTROGRAM_SIZE = (128, 128)
    
    # Model parameters
    TEACHER_DEPTH = 6
    TEACHER_HEADS = 8
    STUDENT_DEPTH = 3
    STUDENT_HEADS = 8
    EMBED_DIM = 256
    PATCH_SIZE = (128, 1)
    
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    LR_DECAY_EPOCHS = 10
    ALPHA = 10  # Weight for L1 loss in stage 3
    
    # Data split
    TRAIN_SPLIT = 0.8
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Paths
    DATA_DIR = "data/CREMA-D"
    MODEL_SAVE_DIR = "models"
    
    # Emotion labels
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']