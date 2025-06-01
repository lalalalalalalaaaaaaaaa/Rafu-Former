import torch
import torch.nn as nn

class CrossVarAttention(nn.Module):
    """
    Cross-variable attention mechanism (treats each variable's entire sequence as a token)
    Core idea:
    1. Compress each variable's time series into a token representation
    2. Compute attention relationships between variables
    """
    def __init__(self, d_model=64, n_heads=8, seq_len=30):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Token generation: using linear layer + average pooling
        self.token_proj = nn.Sequential(
            nn.Linear(1, d_model//2),  # First project to lower dimension
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)  # Then project to target dimension
        )
        
        '''
        self.seq_len = seq_len
        self.value_embedding = nn.Linear(seq_len, d_model)  # Additional projection
        '''
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Input:
            x: [batch_size, seq_len, num_vars] 
            e.g.: [32, 30, 3] means 32 samples, 30 days, 3 variables
        Output:
            new_var_emb: [batch_size, num_vars, d_model] (enhanced variable representations)
            attn_weights: [batch_size, num_vars, num_vars] (attention weight matrix)
        """
        B, T, V = x.shape
        
        # Step 1: Generate variable tokens
        # Original method: encode each time point separately then average (computationally expensive)
        # Optimized method: take sequence mean first then encode (mathematically equivalent but more efficient)
        
        var_means = x.mean(1, keepdim=True)  # [B,1,V]
        var_emb = self.token_proj(var_means.permute(0,2,1))  # [B,V,d_model]
        
        #var_emb = self.value_embedding(x.permute(0,2,1))
        
        # Step 2: Compute cross-variable attention
        # query=key=value=var_emb
        new_var_emb, attn_weights = self.mha(
            var_emb, var_emb, var_emb,
            need_weights=True
        )
        
        # Step 3: Residual connection + layer normalization
        return self.norm(new_var_emb + var_emb), attn_weights