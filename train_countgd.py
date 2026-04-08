#!/usr/bin/env python3
"""
Training script for CountGD with real dataset integration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from countgd_model import LightweightCountGD
from countgd_dataset import get_countgd_dataloaders, download_countgd_dataset
import time

def train_countgd(data_dir='./data', epochs=10, batch_size=8, lr=1e-4):
    """Train CountGD model with real dataset"""
    
    # Setup dataset
    print("Setting up CountGD dataset...")
    download_countgd_dataset(data_dir)
    train_loader, val_loader = get_countgd_dataloaders(data_dir, batch_size)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = LightweightCountGD()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    count_criterion = nn.MSELoss()
    heatmap_criterion = nn.BCELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count_loss_total = 0
        heatmap_loss_total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, texts, counts, heatmaps) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            pred_counts, pred_heatmaps = model(images, texts)
            
            # Calculate losses
            count_loss = count_criterion(pred_counts.squeeze(), counts.squeeze())
            heatmap_loss = heatmap_criterion(pred_heatmaps.squeeze(), heatmaps)
            
            # Combined loss
            loss = count_loss + 0.5 * heatmap_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            count_loss_total += count_loss.item()
            heatmap_loss_total += heatmap_loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        avg_count_loss = count_loss_total / len(train_loader)
        avg_heatmap_loss = heatmap_loss_total / len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Count Loss: {avg_count_loss:.4f}")
        print(f"  Heatmap Loss: {avg_heatmap_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Validation
        if (epoch + 1) % 2 == 0:
            val_loss = validate_model(model, val_loader, count_criterion, heatmap_criterion)
            print(f"  Validation Loss: {val_loss:.4f}")
    
    return model

def validate_model(model, val_loader, count_criterion, heatmap_criterion):
    """Validate model performance"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, texts, counts, heatmaps in val_loader:
            pred_counts, pred_heatmaps = model(images, texts)
            
            count_loss = count_criterion(pred_counts.squeeze(), counts.squeeze())
            heatmap_loss = heatmap_criterion(pred_heatmaps.squeeze(), heatmaps)
            loss = count_loss + 0.5 * heatmap_loss
            
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    print("🚀 Starting CountGD Training with Real Dataset")
    
    # Train model
    model = train_countgd(epochs=5, batch_size=4)
    
    # Save trained model
    torch.save(model.state_dict(), 'countgd_trained.pth')
    print("\n✅ Training completed! Model saved as 'countgd_trained.pth'")
    
    # Quick test
    print("\n🧪 Testing trained model...")
    model.eval()
    test_image = torch.randn(1, 3, 224, 224)
    test_text = ["Count the red objects"]
    
    with torch.no_grad():
        count, heatmap = model(test_image, test_text)
        print(f"Predicted count: {count.item():.2f}")
        print(f"Heatmap shape: {heatmap.shape}")
    
    print("🎉 CountGD integration with real dataset complete!")
