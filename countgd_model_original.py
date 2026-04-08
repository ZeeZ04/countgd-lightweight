import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import timm
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Dict, List
import onnx
import onnxruntime as ort
import torch.nn.functional as F

class LightweightCountGD(nn.Module):
    """Lightweight vision-language model for object counting and grounding with explanations"""
    
    def __init__(self, backbone='mobilevit_s', text_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        
        # Lightweight vision backbone
        self.vision_encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        vision_dim = self.vision_encoder.num_features
        
        # Text encoder (lightweight CLIP)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Fusion layers with configurable attention
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Counting head
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Grounding head with proper spatial processing
        self.spatial_proj = nn.Conv2d(vision_dim, hidden_dim, 1)
        self.grounding_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Store attention weights for explanation
        self.attention_weights = None
        
    def forward(self, image, text, return_explanations=False):
        batch_size = image.size(0)
        
        # Vision features - get both global and spatial
        vision_features_global = self.vision_encoder(image)  # [B, vision_dim]
        
        # Get spatial features from intermediate layers
        with torch.no_grad():
            # Extract features from earlier layers for spatial information
            x = image
            for i, layer in enumerate(self.vision_encoder.children()):
                x = layer(x)
                if i == len(list(self.vision_encoder.children())) - 2:  # Second to last layer
                    spatial_features = x  # [B, C, H, W]
                    break
        
        vision_proj = self.vision_proj(vision_features_global)
        
        # Text features
        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_features = self.clip_model.get_text_features(**text_inputs)
        text_proj = self.text_proj(text_features)
        
        # Cross-attention for explanation
        vision_expanded = vision_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        text_expanded = text_proj.unsqueeze(1)      # [B, 1, hidden_dim]
        
        attended_vision, attention_weights = self.cross_attention(
            vision_expanded, text_expanded, text_expanded
        )
        self.attention_weights = attention_weights
        
        # Fusion
        fused_features = torch.cat([attended_vision.squeeze(1), text_proj], dim=1)
        
        # Count prediction
        count = self.count_head(fused_features)
        
        # Grounding heatmap
        spatial_proj = self.spatial_proj(spatial_features)
        
        # Apply text guidance to spatial features
        text_guidance = text_proj.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim, 1, 1]
        guided_spatial = spatial_proj * text_guidance
        
        heatmap = self.grounding_head(guided_spatial)
        
        if return_explanations:
            explanations = self._generate_explanations(text, attention_weights, heatmap)
            return count, heatmap, explanations
        
        return count, heatmap
    
    def _generate_explanations(self, text_prompts, attention_weights, heatmaps):
        """Generate explanations for object identification"""
        explanations = []
        
        for i, (prompt, attn, heatmap) in enumerate(zip(text_prompts, attention_weights, heatmaps)):
            # Analyze attention patterns
            attn_score = attn.mean().item()
            
            # Analyze heatmap for object locations
            heatmap_np = heatmap.squeeze().detach().cpu().numpy()
            
            # Find peak regions (potential objects)
            threshold = 0.5
            peak_regions = np.where(heatmap_np > threshold)
            
            if len(peak_regions[0]) > 0:
                # Calculate confidence based on peak values
                peak_values = heatmap_np[peak_regions]
                avg_confidence = np.mean(peak_values)
                max_confidence = np.max(peak_values)
                
                # Generate explanation
                explanation = {
                    'prompt': prompt,
                    'attention_score': attn_score,
                    'peak_regions_count': len(peak_regions[0]),
                    'avg_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'explanation_text': self._create_explanation_text(
                        prompt, attn_score, avg_confidence, len(peak_regions[0])
                    )
                }
            else:
                explanation = {
                    'prompt': prompt,
                    'attention_score': attn_score,
                    'peak_regions_count': 0,
                    'avg_confidence': 0.0,
                    'max_confidence': 0.0,
                    'explanation_text': f"No objects matching '{prompt}' were identified with sufficient confidence."
                }
            
            explanations.append(explanation)
        
        return explanations
    
    def _create_explanation_text(self, prompt, attention_score, confidence, region_count):
        """Create human-readable explanation text"""
        
        # Extract object type from prompt
        words = prompt.lower().split()
        object_type = "objects"
        for word in words:
            if word not in ['count', 'the', 'in', 'image', 'how', 'many']:
                object_type = word
                break
        
        explanation = f"Identified {region_count} potential {object_type} regions. "
        
        # Attention explanation
        if attention_score > 0.7:
            explanation += f"Strong text-vision alignment (attention: {attention_score:.2f}) indicates high relevance between prompt and visual features. "
        elif attention_score > 0.4:
            explanation += f"Moderate text-vision alignment (attention: {attention_score:.2f}) suggests partial relevance. "
        else:
            explanation += f"Weak text-vision alignment (attention: {attention_score:.2f}) indicates low confidence. "
        
        # Confidence explanation
        if confidence > 0.8:
            explanation += f"High visual confidence ({confidence:.2f}) in object detection based on learned features."
        elif confidence > 0.5:
            explanation += f"Moderate visual confidence ({confidence:.2f}) in object detection."
        else:
            explanation += f"Low visual confidence ({confidence:.2f}) - objects may be unclear or absent."
        
        return explanation
    
    def inference_with_explanation(self, image, text):
        """Inference method that prints explanations for each identified object"""
        self.eval()
        
        with torch.no_grad():
            count, heatmap, explanations = self.forward(image, text, return_explanations=True)
            
            print(f"\n=== INFERENCE EXPLANATION ===")
            print(f"Input prompt: {text[0] if isinstance(text, list) else text}")
            print(f"Predicted count: {count.item():.2f}")
            
            for i, explanation in enumerate(explanations):
                print(f"\n--- Instance {i+1} Analysis ---")
                print(f"Prompt: {explanation['prompt']}")
                print(f"Regions detected: {explanation['peak_regions_count']}")
                print(f"Explanation: {explanation['explanation_text']}")
                
                # Additional technical details
                print(f"Technical details:")
                print(f"  - Attention score: {explanation['attention_score']:.3f}")
                print(f"  - Average confidence: {explanation['avg_confidence']:.3f}")
                print(f"  - Max confidence: {explanation['max_confidence']:.3f}")
            
            return count, heatmap, explanations

# CountGDDataset moved to countgd_dataset.py

class ModelEvaluator:
    """Evaluation metrics and utilities"""
    
    @staticmethod
    def mean_absolute_error(pred, target):
        return torch.mean(torch.abs(pred - target))
    
    @staticmethod
    def intersection_over_union(pred_mask, target_mask, threshold=0.5):
        pred_binary = (pred_mask > threshold).float()
        target_binary = (target_mask > threshold).float()
        
        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection
        
        return intersection / (union + 1e-8)
    
    @staticmethod
    def measure_inference_time(model, input_tensor, num_runs=100):
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = model(input_tensor, ["test prompt"])
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        return np.mean(times), np.std(times)

class NoiseAugmentation:
    """Visual noise and robustness testing"""
    
    @staticmethod
    def add_gaussian_blur(image, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def adjust_brightness(image, factor=1.2):
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_occlusion(image, num_patches=3, patch_size=50):
        h, w = image.shape[:2]
        occluded = image.copy()
        
        for _ in range(num_patches):
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            occluded[y:y+patch_size, x:x+patch_size] = 0
            
        return occluded

def export_to_onnx(model, dummy_input, output_path):
    """Export model to ONNX format"""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['image', 'text'],
        output_names=['count', 'heatmap']
    )

def train_model(model, dataloader, num_epochs=10, lr=1e-4):
    """Training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    count_criterion = nn.MSELoss()
    heatmap_criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (images, texts, counts, heatmaps) in enumerate(dataloader):
            optimizer.zero_grad()
            
            pred_counts, pred_heatmaps = model(images, texts)
            
            count_loss = count_criterion(pred_counts.squeeze(), counts.squeeze())
            heatmap_loss = heatmap_criterion(pred_heatmaps.squeeze(), heatmaps)
            
            loss = count_loss + 0.5 * heatmap_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def create_countgd_model(size='small'):
    """Factory function to create CountGD models of different sizes
    
    Args:
        size: 'full', 'small', 'tiny', or 'micro'
    """
    from model_configs import get_model_config
    
    config = get_model_config(size)
    
    if size == 'micro':
        from countgd_model_tiny import MicroCountGD
        return MicroCountGD()
    elif size == 'tiny':
        from countgd_model_tiny import TinyCountGD
        return TinyCountGD()
    else:
        return LightweightCountGD(
            backbone=config['backbone'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads']
        )

def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

if __name__ == "__main__":
    # Test different model sizes
    sizes = ['full', 'small', 'tiny', 'micro']
    
    print("CountGD Model Size Comparison:")
    print("=" * 50)
    
    for size in sizes:
        try:
            model = create_countgd_model(size)
            size_mb = get_model_size_mb(model)
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"{size.upper():>6}: {size_mb:>6.1f} MB ({param_count:>9,} params)")
        except Exception as e:
            print(f"{size.upper():>6}: Error - {e}")
    
    # Test inference
    print(f"\nInference Test:")
    print("-" * 30)
    
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = ["count red cars"]
    
    for size in ['small', 'tiny']:
        try:
            model = create_countgd_model(size)
            model.eval()
            
            with torch.no_grad():
                count, heatmap = model(dummy_image, dummy_text)
                print(f"{size.upper():>6}: Count={count.item():.2f}, Heatmap={heatmap.shape}")
        except Exception as e:
            print(f"{size.upper():>6}: Error - {e}")
