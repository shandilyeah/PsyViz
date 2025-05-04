import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import TeacherNetwork
from config import Config
import matplotlib.pyplot as plt
from pathlib import Path

class AudioDemoDataset(Dataset):
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, Config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS
        )(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        # Resize to fixed size
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0),
            size=Config.SPECTROGRAM_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return mel_spec, self.audio_files[idx]

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create demo directory if it doesn't exist
    demo_dir = Path("data/demo")
    # demo_dir = Path("data/CREMA-D/AudioWAV")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if there are any audio files
    if not any(f.endswith('.wav') for f in os.listdir(demo_dir)):
        print(f"No .wav files found in {demo_dir}. Please add some audio files to this directory.")
        return
    
    # Create dataset and dataloader
    dataset = AudioDemoDataset(demo_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = TeacherNetwork().to(device)
    checkpoint_path = Path("models/teacher_best.pth")
    
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("\nProcessing audio files...")
    print("-" * 50)
    
    # Process each audio file
    with torch.no_grad():
        for n, (mel_spec, filename) in enumerate(dataloader):
            mel_spec = mel_spec.to(device)
            
            # Get predictions
            logits = model(mel_spec)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predicted emotion and confidence
            pred_idx = torch.argmax(probabilities, dim=1).item()
            pred_emotion = Config.EMOTION_LABELS[pred_idx]
            confidence = probabilities[0, pred_idx].item()
            
            # Print results
            print(f"\nFile: {filename[0]}")
            print(f"Predicted Emotion: {pred_emotion} (Confidence: {confidence:.2%})")
            print("\nClass Probabilities:")
            for i, emotion in enumerate(Config.EMOTION_LABELS):
                print(f"{emotion}: {probabilities[0, i].item():.2%}")
            print("-" * 50)
            if n > 50:
                break

if __name__ == "__main__":
    main() 