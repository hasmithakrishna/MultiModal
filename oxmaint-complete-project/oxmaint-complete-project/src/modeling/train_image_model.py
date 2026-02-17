#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from sklearn.metrics import classification_report
from loguru import logger
class ImageDataset(Dataset):
    def __init__(self, X, y):
        if len(X.shape) == 3:
            X = X[..., np.newaxis]
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # (N, C, H, W)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DefectCNN(nn.Module):
    """CNN for defect detection"""
    def __init__(self, num_classes=2, dropout=0.5):
        super(DefectCNN, self).__init__()        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)   
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)        
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class ImageModelTrainer:
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)  
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        if total == 0:
            return 0.0, 0.0
        return total_loss / max(len(train_loader), 1), 100 * correct / total
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)  
                if not torch.isnan(loss):
                    total_loss += loss.item()  
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        if total == 0:
            return 0.0, 0.0, [], [], []
        return total_loss / max(len(test_loader), 1), 100 * correct / total, all_preds, all_labels, all_probs
    
    def train(self, train_loader, test_loader, epochs=20, early_stopping_patience=5):
        best_test_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc, _, _, _ = self.evaluate(test_loader)
            self.scheduler.step(test_loss)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        return history
def main():
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "data" / "processed" / "images"
    model_save_path = base_dir / "models" / "image_model"
    model_save_path.mkdir(parents=True, exist_ok=True)
    if not (data_path / "X_train.npy").exists():
        logger.error("Processed image data not found!")
        logger.info("Please run: python src/preprocessing/image_preprocessing.py")
        return
    logger.info("Loading processed image data...")
    X_train = np.load(data_path / "X_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_test = np.load(data_path / "y_test.npy")
    logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    train_dataset = ImageDataset(X_train, y_train)
    test_dataset = ImageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = len(np.unique(y_train))
    logger.info(f"Model config: num_classes={num_classes}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    model = DefectCNN(num_classes=num_classes, dropout=0.5)
    trainer = ImageModelTrainer(model, device=device, learning_rate=0.001)
    logger.info("Starting training...")
    history = trainer.train(train_loader, test_loader, epochs=20, early_stopping_patience=5)
    logger.info("Final evaluation...")
    test_loss, test_acc, y_pred, y_true, y_probs = trainer.evaluate(test_loader)
    logger.success(f"Final Test Accuracy: {test_acc:.2f}%")
    if len(y_true) > 0:
        logger.info("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['OK', 'Defect'], zero_division=0))    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'dropout': 0.5
    }, model_save_path / 'image_model.pth')
    with open(model_save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(model_save_path / 'metadata.json', 'w') as f:
        json.dump({
            'model_type': 'DefectCNN',
            'num_classes': num_classes,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'dropout': 0.5
        }, f, indent=2)
    logger.success(f"Model saved to {model_save_path}")
if __name__ == "__main__":
    main()