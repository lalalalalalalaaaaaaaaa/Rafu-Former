import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBasisGenerator(nn.Module):
    """
    Dynamic basis generator (trend/seasonal/residual)
    Core idea:
    1. Generate three time bases for each variable
    2. Determine basis weights based on cross-variable attention
    """
    def __init__(self, d_model=64, seq_len=30, use_attention=True):
        super().__init__()
        self.seq_len = seq_len
        self.use_attention = use_attention  # New: whether to use attention weights
        
        # Trend basis generation (moving average)
        self.trend_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        
        # Seasonal basis generation (FFT frequency analysis)
        # No trainable parameters needed, directly uses FFT
        
        # Residual smoothing
        self.residual_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        
        # Dynamic weight generator
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Output 3 weights (trend/seasonal/residual)
            nn.Softmax(dim=-1)
        )
        # Traditional mode weight generation
        self.global_weights = nn.Parameter(torch.ones(1, 1, 3)) # [1,1,3] learnable global weights
        
        # Basis representation enhancement
        self.basis_enhancer = nn.Linear(1, d_model)

    def forward(self, x, new_var_emb=None):
        """ 
        Input:
            x: [batch, seq_len, num_vars] (normalized original sequence)
            new_var_emb: [batch, num_vars, d_model] (new representations from attention layer)
        Output:
            enhanced_x: [batch, seq_len, num_vars, d_model] (enhanced temporal representations)
            basis_weights: [batch, num_vars, 3] (basis weights for each variable)
        """
        B, T, V = x.shape
        device = x.device
        
        # === 1. Generate three bases ===
        # Merge batch and variable dimensions for batch processing [B*V,1,T]
        x_conv = x.permute(0,2,1).reshape(-1, 1, T)
        
        # Trend basis (moving average) [B*V,1,T]
        trend = self.trend_conv(x_conv)
        
        # Seasonal basis (FFT reconstruction) [B*V,1,T]
        fft = torch.fft.rfft(x_conv, dim=-1)
        seasonal = torch.fft.irfft(
            fft.abs() * torch.exp(1j * fft.angle()),
            n=T
        )
        
        # Residual basis [B*V,1,T]
        residual = self.residual_conv(x_conv - trend - seasonal)
        
        # Restore shape and merge [B,V,3,T] -> [B,T,V,3]
        bases = torch.stack([trend, seasonal, residual], dim=1)
        bases = bases.view(B,V,3,T).permute(0,3,1,2)

        # === 2. Weight calculation ===
        if self.use_attention:
            weights = self.weight_mlp(new_var_emb)  # [B,V,3]
        else:
            # Traditional mode: use global learnable weights
            weights = torch.softmax(self.global_weights, dim=-1)  # [1,1,3]
            weights = weights.expand(B, V, 3)  # Broadcast to [B,V,3]

        # === 3. Representation enhancement ===
        # Enhance bases from scalar to d_model dimensional vectors [B,T,V,3,1] -> [B,T,V,3,d_model]
        enhanced = self.basis_enhancer(bases.unsqueeze(-1)).squeeze(-2)
        
        # === 4. Weighted fusion ===
        # weights: [B,V,3] -> [B,1,V,3,1] for broadcast multiplication
        weighted = (enhanced * weights.unsqueeze(1).unsqueeze(-1)).sum(-2)
        
        return weighted, weights, bases  # [B,T,V,d_model], [B,V,3]