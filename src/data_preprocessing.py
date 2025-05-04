import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import torchaudio.transforms as T
from config import Config

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None, is_train=True):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        
        # Audio augmentation pipeline
        if is_train and transform is None:
            self.transform = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
            ])
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            win_length=Config.WIN_LENGTH,
            n_mels=Config.N_MELS
        )
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        # Pad or truncate to fixed duration
        target_length = Config.SAMPLE_RATE * Config.DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Apply augmentations if training
        if self.is_train and self.transform:
            audio = self.transform(audio, sample_rate=Config.SAMPLE_RATE)
        
        # Convert to mel spectrogram
        audio_tensor = torch.from_numpy(audio).float()
        mel_spec = self.mel_transform(audio_tensor)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # Reshape to 2D (channels, height, width)
        mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension
        
        # Resize to target size
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0),  # Add batch dimension
            size=Config.SPECTROGRAM_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        
        # Get label
        label = self.labels[idx]
        
        return mel_spec, label

def create_data_loaders(data_dir, batch_size=Config.BATCH_SIZE):
    # Get all audio files and their labels
    audio_files = []
    labels = []
    
    # Map emotion codes to labels
    emotion_map = {
        'ANG': Config.EMOTION_LABELS.index('angry'),
        'DIS': Config.EMOTION_LABELS.index('disgust'),
        'FEA': Config.EMOTION_LABELS.index('fear'),
        'HAP': Config.EMOTION_LABELS.index('happy'),
        'NEU': Config.EMOTION_LABELS.index('neutral'),
        'SAD': Config.EMOTION_LABELS.index('sad')
    }
    
    # Get all WAV files from AudioWAV directory
    audio_dir = os.path.join(data_dir, 'AudioWAV')
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            # Extract emotion from filename (e.g., 1091_IEO_ANG_LO.wav -> ANG)
            emotion = file.split('_')[2]
            if emotion in emotion_map:
                audio_files.append(os.path.join(audio_dir, file))
                labels.append(emotion_map[emotion])
    
    if not audio_files:
        raise ValueError("No valid audio files found in the dataset")
    
    # Split into train and test
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, train_size=Config.TRAIN_SPLIT, random_state=42
    )
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, is_train=True)
    test_dataset = AudioDataset(test_files, test_labels, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader 