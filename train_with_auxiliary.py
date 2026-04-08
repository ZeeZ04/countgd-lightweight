#!/usr/bin/env python3
"""
Training with auxiliary loss (MLE-STAR enhancement)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from countgd_model_optimized import OptimizedCountGD
from countgd_dataset import CountGDDataset

def train_with_auxiliary_loss():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedCountGD().to(device)
    
    train_dataset = CountGDDataset('./data', split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    count_loss_fn = nn.MSELoss()
    
    model.train()
    for epoch in range(10):
        for batch_idx, (images, texts, counts, _) in enumerate(train_loader):
            images, counts = images.to(device), counts.to(device)
            
            # Forward pass
            pred_counts, pred_heatmaps = model(images, texts)
            
            # Main counting loss
            main_loss = count_loss_fn(pred_counts, counts.float())
            
            # Auxiliary loss (from MLE-STAR enhancement)
            aux_counts = model.aux_count_head(
                torch.cat([
                    model.vision_proj(model.vision_encoder(images).mean(dim=[2,3])),
                    model.text_proj(model.text_encoder(**model.tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True
                    )).last_hidden_state.mean(dim=1))
                ], dim=1)
            )
            aux_loss = count_loss_fn(aux_counts.squeeze(), counts.float())
            
            # Combined loss
            total_loss = main_loss + 0.3 * aux_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Main: {main_loss:.4f}, Aux: {aux_loss:.4f}')
    
    torch.save(model.state_dict(), 'countgd_optimized_aux.pth')

if __name__ == "__main__":
    train_with_auxiliary_loss()
