"""Model configuration presets for different size requirements"""

MODEL_CONFIGS = {
    'full': {
        'backbone': 'mobilevit_s',
        'hidden_dim': 256,
        'num_heads': 8,
        'description': 'Full model with best accuracy'
    },
    'small': {
        'backbone': 'mobilevit_xs', 
        'hidden_dim': 128,
        'num_heads': 4,
        'description': 'Balanced size/accuracy'
    },
    'tiny': {
        'backbone': 'mobilevit_xs',
        'hidden_dim': 64,
        'num_heads': 1,
        'description': 'Minimal size, basic accuracy'
    },
    'micro': {
        'backbone': None,  # Custom minimal backbone
        'hidden_dim': 32,
        'num_heads': 1,
        'description': 'Ultra-minimal for edge devices'
    }
}

def get_model_config(size='small'):
    """Get configuration for specified model size"""
    return MODEL_CONFIGS.get(size, MODEL_CONFIGS['small'])
