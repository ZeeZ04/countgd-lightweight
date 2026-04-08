#!/usr/bin/env python3
"""
Minimal training script for MLE-STAR optimized CountGD model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from countgd_model_optimized import OptimizedCountGD
from countgd_dataset import CountGDDataset

def train_optimized_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedCountGD().to(device)
    
    # Dataset
    train_dataset = CountGDDataset('./data', split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    count_loss_fn = nn.MSELoss()
    grounding_loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (images, texts, counts, boxes) in enumerate(train_loader):
            images, counts = images.to(device), counts.to(device)
            
            # Forward pass
            pred_counts, pred_heatmaps = model(images, texts)
            
            # Losses
            count_loss = count_loss_fn(pred_counts, counts.float())
            
            # Simple heatmap target (you may need to adjust this)
            heatmap_targets = torch.zeros_like(pred_heatmaps)
            grounding_loss = grounding_loss_fn(pred_heatmaps, heatmap_targets)
            
            # Total loss
            loss = count_loss + 0.5 * grounding_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} completed. Avg Loss: {total_loss/len(train_loader):.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'countgd_optimized_trained.pth')
    print("Model saved to countgd_optimized_trained.pth")

if __name__ == "__main__":
    train_optimized_model()
