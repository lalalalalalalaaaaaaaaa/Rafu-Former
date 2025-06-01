import torch
import torch.nn as nn
from model import RafuFormer
from data_provider.data_factory import data_provider
import argparse
import os


def create_block_mask(shape, pred_len, device='cuda'):
    """Generate dynamic masking pattern based on prediction length"""
    B, T, V = shape
    ratio = min(0.4, 0.15 + pred_len/2000)  # Dynamic ratio calculation
    
    if pred_len <= 192:  # Short-term prediction: random masking
        mask = (torch.rand(B, T, V, device=device) < ratio)
    else:  # Long-term prediction: continuous block masking
        block_size = int(T * ratio)
        starts = torch.randint(0, T-block_size, (B,V), device=device)
        mask = (torch.arange(T,device=device).view(1,T,1) >= starts.unsqueeze(1)) & \
               (torch.arange(T,device=device).view(1,T,1) < (starts+block_size).unsqueeze(1))
        # Debug output (optional)
    #print(f"pred_len={pred_len} Actual masking ratio:{mask.float().mean().item():.1%}")
    return mask.float()

def pretrain(args):
    """Pretraining function with dynamic masking"""
    model = RafuFormer(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        load_pretrain=False
    ).to(args.device)

    train_loader = data_provider(args, "train")[1]
    
    # Lightweight optimizer configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    criterion = nn.MSELoss()

    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y, _, _, _ in train_loader:
            batch_x, batch_y = batch_x.float().to(args.device), batch_y.float().to(args.device)
            
            # Dynamic masking (based on current pred_len)
            block_mask = (create_block_mask(batch_x.shape, args.pred_len, device=args.device) > 0)  # Modified
            context = batch_x.masked_fill(block_mask, 0)
            
            
            # Only predict target length (simplified computation)
            future_pred = model(context, None, pretrain_mode=True)
            loss = criterion(future_pred[:, :args.pred_len], batch_y[:, -args.pred_len:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Pretrain Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    
    # Save weights
    os.makedirs(os.path.join('records', args.data_path.split('.')[0], 'features_M',
                           f'seq_len{args.seq_len},pred_len{args.pred_len}', 'checkpoint'), exist_ok=True)
    
    torch.save( model.state_dict() , os.path.join('records', args.data_path.split('.')[0], 'features_M',
                   f'seq_len{args.seq_len},pred_len{args.pred_len}', 
                   'checkpoint', 'pretrain_weights.pth'))

if __name__ == "__main__":
    # Pretraining specific parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='custom', required=True)  # Dataset type
    parser.add_argument('--pretrain_epochs', type=int, default=12)
    parser.add_argument('--pretrain_lr', type=float, default=5e-4)
    # Inherited parameters from main.py
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--label_len', type=int, default=96, help='start token length')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                            'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model') 
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    args = parser.parse_args()
    pretrain(args)