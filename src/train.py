import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_preprocessing import create_data_loaders
from models import TeacherNetwork, StudentNetwork
from config import Config

def train_teacher(train_loader, val_loader):
    print("Training Teacher Network...")
    teacher = TeacherNetwork().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_DECAY_EPOCHS, gamma=0.5)
    
    best_val_acc = 0
    writer = SummaryWriter('runs/teacher')
    
    for epoch in range(Config.NUM_EPOCHS):
        # Training
        teacher.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}'):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        teacher.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = teacher(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(teacher.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, 'teacher_best.pth'))
    
    writer.close()
    return teacher

def train_student_feature_matching(teacher, train_loader, val_loader):
    print("\nTraining Student Network (Feature Matching)...")
    student = StudentNetwork().to(Config.DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(student.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_DECAY_EPOCHS, gamma=0.5)
    
    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
    
    writer = SummaryWriter('runs/student_feature_matching')
    
    for epoch in range(Config.NUM_EPOCHS):
        # Training
        student.train()
        train_loss = 0
        
        for inputs, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}'):
            inputs = inputs.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Get teacher features
            with torch.no_grad():
                teacher_features = teacher.vit_encoder(teacher.coord_encoding(teacher.conv_stem(inputs)))
            
            # Get student features
            student_features = student.vit_encoder(student.conv_stem(inputs))
            
            # Compute L1 loss between features
            loss = criterion(student_features, teacher_features)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}')
    
    writer.close()
    return student

def train_student_final(student, train_loader, val_loader):
    print("\nTraining Student Network (Final Stage)...")
    criterion_ce = nn.CrossEntropyLoss()
    criterion_l1 = nn.L1Loss()
    optimizer = optim.Adam(student.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_DECAY_EPOCHS, gamma=0.5)
    
    best_val_acc = 0
    writer = SummaryWriter('runs/student_final')
    
    for epoch in range(Config.NUM_EPOCHS):
        # Training
        student.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}'):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Get student features and outputs
            student_features = student.vit_encoder(student.conv_stem(inputs))
            outputs = student.classifier(student_features.mean(dim=1))
            
            # Compute combined loss
            loss_ce = criterion_ce(outputs, labels)
            loss_l1 = criterion_l1(student_features, student_features.detach())  # Self-similarity
            loss = loss_ce + Config.ALPHA * loss_l1
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        student.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = student(inputs)
                loss = criterion_ce(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, 'student_best.pth'))
    
    writer.close()
    return student

def main():
    # Create directories
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Load data
    train_loader, val_loader = create_data_loaders(Config.DATA_DIR)
    
    # Stage 1: Train Teacher
    if not os.path.exists(os.path.join(Config.MODEL_SAVE_DIR, 'teacher_best.pth')):
        print(f"Training teacher from scratch")
        teacher = train_teacher(train_loader, val_loader)
    else:
        print(f"Loading teacher from {Config.MODEL_SAVE_DIR}/teacher_best.pth")
        teacher = TeacherNetwork().to(Config.DEVICE)
        teacher.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_DIR, 'teacher_best.pth'), map_location=Config.DEVICE, weights_only=True))
    
    # Stage 2: Feature Matching
    student = train_student_feature_matching(teacher, train_loader, val_loader)
    
    # Stage 3: Final Training
    student = train_student_final(student, train_loader, val_loader)

if __name__ == '__main__':
    main() 