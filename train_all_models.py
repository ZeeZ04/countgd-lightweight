#!/usr/bin/env python3
"""Train CountGD models of different sizes"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from countgd_model import create_countgd_model, get_model_size_mb
from countgd_dataset import get_countgd_dataloaders, download_countgd_dataset

def train_model(model, dataloader, epochs=5, lr=1e-4, model_name="model"):
    """Simple training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    count_criterion = nn.MSELoss()
    
    model.train()
    print(f"\nTraining {model_name}...")
    print(f"Model size: {get_model_size_mb(model):.1f} MB")
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        for images, texts, counts, heatmaps in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'forward') and 'return_explanations' in model.forward.__code__.co_varnames:
                pred_counts, pred_heatmaps = model(images, texts, return_explanations=False)
            else:
                pred_counts, pred_heatmaps = model(images, texts)
            
            # Loss calculation
            count_loss = count_criterion(pred_counts.squeeze(), counts.squeeze())
            loss = count_loss  # Simplified - just count loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train CountGD models')
    parser.add_argument('--size', choices=['full', 'small', 'tiny', 'micro'], 
                       default='small', help='Model size')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data-path', type=str, default='./data', help='Dataset path')
    parser.add_argument('--train-all', action='store_true', help='Train all model sizes')
    
    args = parser.parse_args()
    
    # Setup dataset
    print("Setting up dataset...")
    data_dir = download_countgd_dataset(args.data_path)
    train_loader, val_loader = get_countgd_dataloaders(data_dir, batch_size=4)
    
    if args.train_all:
        # Train all model sizes
        sizes = ['small', 'tiny', 'micro', 'full']
        
        for size in sizes:
            print(f"\n{'='*50}")
            print(f"Training {size.upper()} model")
            print(f"{'='*50}")
            
            try:
                # Create model
                model = create_countgd_model(size)
                
                # Train model
                trained_model = train_model(
                    model, train_loader, 
                    epochs=args.epochs, 
                    lr=args.lr,
                    model_name=f"{size.upper()}"
                )
                
                # Save model
                save_path = f"countgd_{size}.pth"
                torch.save(trained_model.state_dict(), save_path)
                print(f"✅ Saved {size} model to {save_path}")
                
            except Exception as e:
                print(f"❌ Error training {size} model: {e}")
    
    else:
        # Train single model
        print(f"Training {args.size.upper()} model...")
        
        model = create_countgd_model(args.size)
        trained_model = train_model(
            model, train_loader, 
            epochs=args.epochs, 
            lr=args.lr,
            model_name=args.size.upper()
        )
        
        # Save model
        save_path = f"countgd_{args.size}.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    main()
