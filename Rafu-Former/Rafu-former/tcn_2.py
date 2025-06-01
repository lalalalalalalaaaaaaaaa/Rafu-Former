import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time 

# --- Configuration ---
ROOT_PATH = 'datasets/electricity'
DATA_PATH = 'electricity.csv'
SEQ_LEN = 96
PRED_LEN = 96

# Set the mode to run
RUN_MODE = 'multivariate'

# Define configurations
CONFIGS = {
    'univariate': {
        'FEATURES': 'S',
        'TARGET': 'OT',
        'DESCRIPTION': 'Univariate TCN Regression on Electricity Data',
        'INPUT_SIZE': 1
    },
    'multivariate': {
        'FEATURES': 'M',
        'TARGET': None,
        'DESCRIPTION': 'Multivariate TCN Regression on Electricity Data',
        'INPUT_SIZE': None
    }
}

# Validate and get configuration
if RUN_MODE not in CONFIGS:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}. Choose from {list(CONFIGS.keys())}")
CURRENT_CONFIG = CONFIGS[RUN_MODE]

FEATURES = CURRENT_CONFIG['FEATURES']
TARGET = CURRENT_CONFIG['TARGET']
DESCRIPTION = CURRENT_CONFIG['DESCRIPTION']

# --- Dataset Class ---
class ElectricityDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='OT', scale=True):
        if size is None:
            self.seq_len = 96
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.features == 'M':
            cols_data = [col for col in df_raw.columns if col != 'date']
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target is None or self.target not in df_raw.columns:
                raise ValueError(f"TARGET column '{self.target}' must be specified and exist in data for features='S'.")
            cols_data = [self.target]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features flag: {self.features}")

        num_total = len(df_data)
        num_train = int(num_total * 0.7)
        num_test = int(num_total * 0.2)
        num_vali = num_total - num_train - num_test

        border1s = [0, num_train - self.seq_len - self.pred_len + 1, num_total - num_test - self.seq_len - self.pred_len + 1]
        border2s = [num_train, num_train + num_vali, num_total]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            self.scaler = StandardScaler()
            train_data_raw = df_data.iloc[:num_train].values
            self.scaler.fit(train_data_raw)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1 + self.seq_len:border2 + self.seq_len]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Ensure valid slicing
        if s_end > len(self.data_x) or r_end > len(self.data_y) + self.seq_len:
            raise IndexError("Index out of bounds for sequence extraction")

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin - self.seq_len:r_end - self.seq_len]

        # Verify shapes
        if seq_x.shape[0] != self.seq_len or seq_y.shape[0] != self.pred_len:
            raise ValueError(f"Invalid sequence length: seq_x={seq_x.shape[0]}, seq_y={seq_y.shape[0]}")

        seq_x_tensor = torch.from_numpy(seq_x).float()
        seq_y_tensor = torch.from_numpy(seq_y).float()

        return seq_x_tensor, seq_y_tensor

    def __len__(self):
        # Only count sequences where both seq_x and seq_y can be fully extracted
        max_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return max(0, max_len)

    def inverse_transform(self, data):
        if self.scale and hasattr(self, 'scaler'):
            return self.scaler.inverse_transform(data)
        return data

# --- TCN Model Definition ---
class TCNForForecasting(nn.Module):
    def __init__(self, input_features, output_features, hidden_channels=64, kernel_size=3, dropout=0.2, num_layers=7, pred_len=96):
        super(TCNForForecasting, self).__init__()
        self.pred_len = pred_len
        self.output_features = output_features

        layers = []
        for i in range(num_layers):
            in_ch = input_features if i == 0 else hidden_channels
            out_ch = hidden_channels
            dil = 2 ** i
            padding = (kernel_size - 1) * dil
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dil, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.tcn = nn.Sequential(*layers)
        self.forecasting_head = nn.Linear(hidden_channels, pred_len * output_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)
        last_out = tcn_out[:, :, -1]
        forecast = self.forecasting_head(last_out)
        forecast = forecast.view(forecast.size(0), self.pred_len, self.output_features)
        return forecast

# --- Training and Evaluation Functions ---
def train(model, train_loader, vali_loader, criterion, optimizer, num_epochs, device):
    model.train()
    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_train_loss = train_loss / len(train_loader)
        epoch_end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Train Loss: {epoch_train_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.4f}s")

        if vali_loader is not None:
            vali_mse, vali_mae = evaluate(model, vali_loader, device)
            print(f"  Validation Metrics: MSE = {vali_mse:.4f}, MAE = {vali_mae:.4f}")
            model.train()

def evaluate(model, data_loader, device):
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    total_loss /= len(data_loader)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Calculate MSE and MAE using scikit-learn
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    mae = mean_absolute_error(trues.flatten(), preds.flatten())

    print(f"Evaluation Loss (MSE): {total_loss:.4f}")
    return mse, mae

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Running {DESCRIPTION} ---")

    # Load data
    df_raw_check_size = pd.read_csv(os.path.join(ROOT_PATH, DATA_PATH))
    if FEATURES == 'M':
        cols_data_check_size = [col for col in df_raw_check_size.columns if col != 'date']
        INPUT_SIZE = len(cols_data_check_size)
        OUTPUT_FEATURES = INPUT_SIZE
    elif FEATURES == 'S':
        INPUT_SIZE = 1
        OUTPUT_FEATURES = 1
    else:
        raise ValueError(f"Unsupported features flag: {FEATURES}")

    # Create datasets
    train_dataset = ElectricityDataset(root_path=ROOT_PATH, data_path=DATA_PATH, flag='train', features=FEATURES)
    vali_dataset = ElectricityDataset(root_path=ROOT_PATH, data_path=DATA_PATH, flag='val', features=FEATURES)
    test_dataset = ElectricityDataset(root_path=ROOT_PATH, data_path=DATA_PATH, flag='test', features=FEATURES)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    vali_loader = DataLoader(vali_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNForForecasting(input_features=INPUT_SIZE, output_features=OUTPUT_FEATURES).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    NUM_EPOCHS = 10
    train(model, train_loader, vali_loader, criterion, optimizer, NUM_EPOCHS, device)

    # Test
    print("\nStarting testing...")
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")