import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from models import TeacherNetwork
from config import Config
from data_preprocessing import create_data_loaders
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics_dict, save_path):
    with open(save_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall Accuracy: {metrics_dict['accuracy']:.2%}\n\n")
        
        f.write("Per-class Metrics:\n")
        for i, emotion in enumerate(Config.EMOTION_LABELS):
            f.write(f"\n{emotion}:\n")
            f.write(f"  Precision: {metrics_dict['precision'][i]:.2%}\n")
            f.write(f"  Recall:    {metrics_dict['recall'][i]:.2%}\n")
            f.write(f"  F1 Score:  {metrics_dict['f1'][i]:.2%}\n")
        
        f.write("\nMacro Averages:\n")
        f.write(f"  Precision: {metrics_dict['macro_precision']:.2%}\n")
        f.write(f"  Recall:    {metrics_dict['macro_recall']:.2%}\n")
        f.write(f"  F1 Score:  {metrics_dict['macro_f1']:.2%}\n")
        

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = TeacherNetwork().to(device)
    checkpoint_path = Path("models/teacher_best.pth")
    
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create dataset and dataloader
    train_loader, test_loader = create_data_loaders(Config.DATA_DIR)
    
    # # Concatenate the datasets from both loaders
    # combined_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, test_loader.dataset])
    
    # # Create a new data loader from the combined dataset
    # combined_loader = DataLoader(
    #     combined_dataset,
    #     batch_size=Config.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    # )

    combined_loader = test_loader
    
    print("\nEvaluating model on CREMA-D dataset...")
    
    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Process all samples
    with torch.no_grad():
        for mel_spec, labels in combined_loader:
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits = model(mel_spec)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Compute precision, recall, and F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Compute macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Compute accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
    }
    
    # Save metrics to file
    save_metrics(metrics, results_dir / "metrics.txt")
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        cm,
        Config.EMOTION_LABELS,
        results_dir / "confusion_matrix.png"
    )
    
    print("\nEvaluation complete!")
    print(f"Metrics saved to {results_dir}/metrics.txt")
    print(f"Confusion matrix saved to {results_dir}/confusion_matrix.png")

if __name__ == "__main__":
    main() 