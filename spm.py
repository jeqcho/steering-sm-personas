import torch
import torch.nn as nn
import torch.optim as optim

class SPM(nn.Module):
    def __init__(self, num_layers: int, token_dim: int, num_heads: int, num_virtual_tokens: int = 1):
        """
        Initialize the Steering Persona Model (SPM).
        
        Args:
            num_layers (int): Number of transformer layers (L)
            token_dim (int): Token dimension (D)
            num_heads (int): Number of attention heads (H)
            num_virtual_tokens (int): Number of virtual tokens (T), default is 1
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_virtual_tokens = num_virtual_tokens
        
        # Calculate the output dimension for multi-query attention
        # Shape: [T, L×2×D/H]
        output_dim = num_layers * 2 * (token_dim // num_heads)
        
        # Two-layer MLP with 32 hidden units
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),  # Input is a single scalar
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SPM.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, T, L×2×D/H]
        """
        # Pass through MLP
        output = self.mlp(x)
        
        # Reshape to [batch_size, T, L×2×D/H]
        batch_size = x.shape[0]
        output = output.view(batch_size, self.num_virtual_tokens, -1)
        
        return output

def create_spm_optimizer(model: SPM, learning_rate: float = 0.001, weight_decay: float = 0.001):
    """
    Create the AdamW optimizer for the SPM with specified learning rate and weight decay.
    
    Args:
        model (SPM): The SPM model
        learning_rate (float): Learning rate for AdamW optimizer
        weight_decay (float): Weight decay for AdamW optimizer
        
    Returns:
        optim.AdamW: Configured AdamW optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

