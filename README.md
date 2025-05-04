# Emotion Recognition using Vision Transformers

This project implements an emotion recognition system using Vision Transformers (ViT) to classify emotions from audio spectrograms. The system is trained on the CREMA-D dataset and uses a teacher-student architecture for knowledge distillation.

## Project Structure

```
.
├── data/
│   ├── CREMA-D/          # CREMA-D dataset
│   │   └── AudioWAV/     # Audio files
│   └── demo/             # Directory for demo audio files
├── models/               # Directory for saved model checkpoints
├── results/              # Directory for evaluation results
├── src/
│   ├── config.py         # Configuration parameters
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── models.py         # Model architectures
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── demo.py           # Demo script for inference
└── requirements.txt      # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CREMA-D dataset and place it in the `data/CREMA-D/AudioWAV` directory.

## Usage

### Training

To train the model:
```bash
python src/train.py
```

This will:
- Preprocess the audio files into mel spectrograms
- Train the teacher model
- Train the student model using knowledge distillation
- Save the best model checkpoints to the `models` directory

### Evaluation

To evaluate the model on the test set:
```bash
python src/evaluate.py
```

This will:
- Load the trained teacher model
- Evaluate on the test set
- Save metrics to `results/metrics.txt`
- Generate and save a confusion matrix to `results/confusion_matrix.png`

The evaluation metrics include:
- Overall accuracy
- Per-class precision, recall, and F1 scores
- Macro-averaged metrics

### Demo

To run inference on custom audio files:
```bash
python src/demo.py
```

This will:
- Process all `.wav` files in the `data/demo` directory
- Make predictions using the trained teacher model
- Display predicted emotions and confidence scores

Place your audio files in the `data/demo` directory before running the demo.

## Model Architecture

The system uses a Vision Transformer-based architecture:

1. **Teacher Network**:
   - ConvStem for initial feature extraction
   - Coordinate encoding for spatial information
   - ViT encoder for global feature learning
   - Classification head

2. **Student Network**:
   - Similar architecture to teacher but without coordinate encoding
   - Trained using knowledge distillation from teacher

## Configuration

Key parameters can be adjusted in `src/config.py`:
- Audio processing parameters (sample rate, n_fft, etc.)
- Model architecture parameters (embedding dimension, number of heads, etc.)
- Training parameters (batch size, learning rate, etc.)

## Results

Evaluation results are saved in the `results` directory:
- `metrics.txt`: Detailed performance metrics
- `confusion_matrix.png`: Visualization of the confusion matrix

## License

This project is licensed under the MIT License - see the LICENSE file for details.
