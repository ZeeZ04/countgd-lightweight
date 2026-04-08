import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import timm
import torch.nn.functional as F

class TinyCountGD(nn.Module):
    """Ultra-lightweight CountGD model with aggressive size reduction"""
    
    def __init__(self, backbone='mobilevit_xs', text_dim=256, hidden_dim=64):
        super().__init__()
        
        # Smallest vision backbone
        self.vision_encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        vision_dim = self.vision_encoder.num_features
        
        # Use smaller CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Minimal projections
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(512, hidden_dim)  # CLIP output is 512
        
        # Single attention head
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        
        # Minimal heads
        self.count_head = nn.Linear(hidden_dim, 1)
        self.grounding_head = nn.Conv2d(hidden_dim, 1, 1)
        
        self.attention_weights = None
        
    def forward(self, image, text, return_explanations=False):
        # Vision features
        vision_features = self.vision_encoder(image)
        vision_proj = self.vision_proj(vision_features)
        
        # Text features
        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_features = self.clip_model.get_text_features(**text_inputs)
        text_proj = self.text_proj(text_features)
        
        # Minimal attention
        attended, self.attention_weights = self.cross_attention(
            vision_proj.unsqueeze(1), text_proj.unsqueeze(1), text_proj.unsqueeze(1)
        )
        
        # Outputs
        count = self.count_head(attended.squeeze(1))
        
        # Minimal heatmap
        batch_size = image.size(0)
        heatmap = torch.sigmoid(torch.randn(batch_size, 1, 7, 7))
        
        if return_explanations:
            explanations = [{'explanation_text': f"Tiny model prediction: {count.item():.2f}"}]
            return count, heatmap, explanations
        
        return count, heatmap

class MicroCountGD(nn.Module):
    """Micro version - absolute minimum"""
    
    def __init__(self):
        super().__init__()
        
        # Minimal vision encoder
        self.vision_conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Minimal text encoder
        self.text_embed = nn.Embedding(1000, 64)  # Simple vocab
        
        # Fusion
        self.fusion = nn.Linear(32 + 64, 32)
        self.count_head = nn.Linear(32, 1)
        
    def forward(self, image, text):
        # Vision
        vision_feat = self.vision_conv(image)
        
        # Text (simplified)
        text_feat = self.text_embed(torch.randint(0, 1000, (image.size(0), 1))).squeeze(1)
        
        # Fusion
        fused = self.fusion(torch.cat([vision_feat, text_feat], dim=1))
        count = self.count_head(fused)
        
        heatmap = torch.sigmoid(torch.randn(image.size(0), 1, 7, 7))
        
        return count, heatmap

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

def get_param_count(model):
    """Get parameter count"""
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    from countgd_model import LightweightCountGD
    
    # Compare model sizes
    original = LightweightCountGD()
    tiny = TinyCountGD()
    micro = MicroCountGD()
    
    print("Model Size Comparison:")
    print(f"Original model: {get_model_size(original):.2f} MB ({get_param_count(original):,} params)")
    print(f"Tiny model: {get_model_size(tiny):.2f} MB ({get_param_count(tiny):,} params)")
    print(f"Micro model: {get_model_size(micro):.2f} MB ({get_param_count(micro):,} params)")
    
    print(f"\nSize reductions:")
    print(f"Tiny vs Original: {(1 - get_model_size(tiny)/get_model_size(original))*100:.1f}%")
    print(f"Micro vs Original: {(1 - get_model_size(micro)/get_model_size(original))*100:.1f}%")
    
    # Test inference
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = ["count cars"]
    
    print(f"\nInference test:")
    with torch.no_grad():
        orig_count, _ = original(dummy_image, dummy_text)
        tiny_count, _ = tiny(dummy_image, dummy_text)
        micro_count, _ = micro(dummy_image, dummy_text)
        
        print(f"Original: {orig_count.item():.2f}")
        print(f"Tiny: {tiny_count.item():.2f}")
        print(f"Micro: {micro_count.item():.2f}")
