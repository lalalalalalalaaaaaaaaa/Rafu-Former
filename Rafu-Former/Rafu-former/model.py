import torch
import torch.nn as nn
from layers.attention import CrossVarAttention
from layers.decomposition import DynamicBasisGenerator
import os

class RafuFormer(nn.Module):
    """
    Complete model architecture:
    1. Cross-variable attention
    2. Dynamic basis decomposition
    3. Variable-independent prediction
    """
    def __init__(self, seq_len, pred_len, d_model=64, n_heads=8, device='cuda', use_attention=True, load_pretrain=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device
        self.use_attention = use_attention  # Whether to use variable attention

        # New pretrained weights loading logic
        if load_pretrain:
            self.load_pretrained_weights()

        # Improved reconstruction network
        self.recon_proj = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, 1)
        )

        # Add future prediction head
        self.future_head = nn.Linear(d_model, pred_len)    

        # Only add this line - reconstruction projection layer
        self.recon_proj = nn.Linear(d_model, 1)  # Project d_model dimension back to original value    

        # 1. Cross-variable attention
        #self.var_attention = CrossVarAttention(d_model, n_heads, seq_len)
        self.var_attention = CrossVarAttention(d_model, n_heads)
        
        # 2. Dynamic basis generation
        self.basis_gen = DynamicBasisGenerator(d_model, seq_len, use_attention=self.use_attention)

        # Traditional basis fusion (MLP learns weights)
        if not self.use_attention:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3, 16),  # Input weights for 3 bases
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1)
            )
        
        # 3. Predictor (partially shared parameters)
        self.shared_encoder = nn.Sequential(
            nn.Linear(seq_len * d_model, 256),
            nn.ReLU()
        )
        # Independent prediction head for each variable
        self.var_predictors = nn.ModuleList([
            nn.Linear(256, pred_len) for _ in range(400)  # Assuming max 10 variables
        ])
        
        # 4. Loss functions
        self.criterion_mse = nn.MSELoss()
        self.criterion_smooth = nn.L1Loss()

    def forward(self, x, index, y=None, train=True, y_mark=None, pretrain_mode=False):
        B, T, V = x.shape
        #if V > len(self.var_predictors):
        #    raise ValueError(f"Number of variables {V} exceeds preset maximum of 10")
        
        # === 1. Data normalization ===
        mean = x.mean(1, keepdim=True)  # [B,1,V]
        std = x.std(1, keepdim=True) + 1e-5  # [B,1,V]
        x_norm = (x - mean) / std  # [B,T,V]

        if pretrain_mode:
            # Use same path as formal training
            var_emb, _ = self.var_attention(x_norm)
            adjusted_x, _, _ = self.basis_gen(x_norm, var_emb)
            
            # Reconstruction task
            #recon = self.recon_proj(adjusted_x).squeeze(-1)
            
            # Prediction task (same structure as formal training)
            feat = adjusted_x.permute(0,2,1,3).reshape(B,V,-1)
            shared = self.shared_encoder(feat)
            future_pred = torch.stack([
                self.var_predictors[i](shared[:,i]) 
                for i in range(V)], dim=2)
            
           # return recon, future_pred
            return future_pred

        if self.use_attention:
            # === 2. Variable attention ===
            new_var_emb, attn_weights = self.var_attention(x_norm)  # [B,V,d_model], [B,V,V]
            
            # === 3. Dynamic basis generation ===
            adjusted_x, basis_weights, bases = self.basis_gen(x_norm, new_var_emb)  # [B,T,V,d_model], [B,V,3]
        else:
            # Traditional fusion path
            adjusted_x, basis_weights, bases = self.basis_gen(x)  # [B,T,V,3,d_model]
            #basis_weights = self.fusion_mlp(raw_weights)  # [B,V,3]
            #adjusted_x = (bases * basis_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=-2)
        
        # === 4. Prediction ===
        # Adjust dimensions [B,V,T*d_model]
        feat = adjusted_x.permute(0,2,1,3).reshape(B,V,-1)
        
        # Shared encoding
        shared = self.shared_encoder(feat)  # [B,V,256]
        
        # Independent prediction
        pred = torch.stack([
            self.var_predictors[i](shared[:,i]) 
            for i in range(V)], dim=2)  # [B,pred_len,V]
        
        # Denormalization
        std = std.expand(-1, self.pred_len, -1)  # [32,96,321]
        mean = mean.expand(-1, self.pred_len, -1) # [32,96,321]
        #print("Pred shape:", pred.shape)
        #print("Std shape:", std.shape)
        #print("Mean shape:", mean.shape)
        pred = pred * std + mean
        
        if train:
            loss_pred = self.criterion_mse(pred, y)
            loss_smooth = self.criterion_smooth(
                basis_weights[:,:,1:],  # Ensure smooth weight changes
                basis_weights[:,:,:-1]
            )
            return pred, loss_pred, loss_smooth, adjusted_x, basis_weights, bases
        else:
            return pred, adjusted_x, basis_weights, bases
    
    def load_pretrained_weights(self):
        pretrain_path = os.path.join('records','electricity','features_M',
                              'seq_len'+str(self.seq_len)+','+'pred_len'+str(self.pred_len),'checkpoint','pretrain_weights.pth')
        try:
            pretrain_dict = torch.load(pretrain_path)
            model_dict = self.state_dict()
            #print("Keys in weight file 1:", pretrain_dict.keys())
            # 1. Filter mismatched keys
            pretrain_dict = {k: v for k, v in pretrain_dict.items() 
                            if k in model_dict and v.shape == model_dict[k].shape}
            #print("Keys in weight file 2:", pretrain_dict.keys())
            # 2. Update current model parameters
            model_dict.update(pretrain_dict)
            print("Example parameter names:", list(model_dict.keys())[:5])  # Print first 5 parameter names

            # 3. Strict loading
            self.load_state_dict(model_dict)
            
            # 4. Print loading info
            print(f"Successfully loaded {len(pretrain_dict)}/{len(model_dict)} parameters")
        except:
            print("Error during loading")