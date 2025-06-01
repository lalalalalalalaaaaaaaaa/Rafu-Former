import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import itertools

# --- Configuration ---
ROOT_PATH = 'datasets/electricity' 
DATA_PATH = 'electricity.csv'
SEQ_LEN = 96 # Keeping SEQ_LEN fixed as in Basisformer experiments

# --- Prediction Lengths to Test ---
PRED_LENS_TO_TEST = [96, 192, 336, 720]

# --- Set the mode to run ---
# Choose 'univariate' or 'multivariate'
#RUN_MODE = 'univariate'
RUN_MODE = 'multivariate' # Uncomment this line and comment the one above to run multivariate mode

# Define configurations for different modes
CONFIGS = {
    'univariate': {
        'FEATURES': 'S',
        'TARGET': 'OT', # Specify the target column for univariate mode
        'BASE_DESCRIPTION': 'Univariate GRU Regression on Electricity Data (Pred Len: {})',
        'INPUT_SIZE': 1 # For univariate, input size is 1 feature
    },
    'multivariate': {
        'FEATURES': 'M',
        'TARGET': None, # Target is determined by FEATURES='M' in load function
        'BASE_DESCRIPTION': 'Multivariate GRU Regression on Electricity Data (Pred Len: {})',
        # INPUT_SIZE will be determined dynamically based on data columns
        'INPUT_SIZE': None
    }
}

# Get the configuration for the selected mode
if RUN_MODE not in CONFIGS:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}. Choose from {list(CONFIGS.keys())}")
CURRENT_CONFIG = CONFIGS[RUN_MODE]

FEATURES = CURRENT_CONFIG['FEATURES']
TARGET = CURRENT_CONFIG['TARGET']
INPUT_SIZE = CURRENT_CONFIG['INPUT_SIZE'] # Will be updated after loading data for multivariate mode
BASE_DESCRIPTION = CURRENT_CONFIG['BASE_DESCRIPTION']

# --- Metric Functions (Copied from evaluate_tool.py) ---
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

# --- Dataset Class (Adapted from Dataset_Custom logic) ---

class ElectricityDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='OT', scale=True, timeenc=0, freq='h',
                 scaler=None):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = SEQ_LEN # Using global SEQ_LEN
            self.label_len = 0 # Simplified label_len for this baseline
            self.pred_len = 96 # Default, will be overridden by size in __main__
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.external_scaler = scaler 
        self.__read_data__()
        self.len = self.__len__()


    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Identify data columns based on FEATURES flag
        if self.features == 'M':
            cols_data = [col for col in df_raw.columns if col != 'date']
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target is None or self.target not in df_raw.columns:
                 raise ValueError(f"TARGET column '{self.target}' must be specified and exist in data for features='S'.")
            cols_data = [self.target]
            df_data = df_raw[[self.target]]
        else:
             raise ValueError(f"Unsupported features flag for this baseline: {self.features}. Use 'S' or 'M'.")


        # Define train, val, test borders based on 70/20/10 split 
        num_total = len(df_data)
        num_train = int(num_total * 0.7)
        num_test = int(num_total * 0.2)
        num_vali = num_total - num_train - num_test

        border1s = [0, num_train, num_total - num_test]
        border2s = [num_train, num_train + num_vali, num_total]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            if self.external_scaler is not None:
                self.scaler = self.external_scaler 
                data = self.scaler.transform(df_data.values)
            elif self.set_type == 0:
                self.scaler = StandardScaler()
                train_data_raw = df_data.iloc[border1s[0]:border2s[0]].values
                self.scaler.fit(train_data_raw)
                data = self.scaler.transform(df_data.values)
            else:
                 raise RuntimeError("Scaler not provided for validation/test set.")
        else:
            data = df_data.values


        # Select the data for the current split after standardization
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # Target sequence slice is pred_len after seq_len
        seq_x = self.data_x[s_begin:s_end] # Input sequence (seq_len, n_features_original)
        seq_y_full_slice = self.data_y[s_end : s_end + self.pred_len] # Target sequence (pred_len, n_features_original)


        # Select output features based on FEATURES flag
        if self.features == 'S':
             seq_y = seq_y_full_slice[:, 0:1] # Target is the single (0th) feature
        elif self.features == 'M':
             seq_y = seq_y_full_slice # Target is all features
        else:
             raise ValueError(f"Unsupported features flag in __getitem__: {self.features}")


        # Convert to tensors
        seq_x_tensor = torch.from_numpy(seq_x).float()
        seq_y_tensor = torch.from_numpy(seq_y).float()

        return seq_x_tensor, seq_y_tensor # Return shapes (seq_len, n_features_in), (pred_len, n_features_out)


    def __len__(self):
         return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse transform standardized data."""
        if self.scale and hasattr(self, 'scaler') and self.scaler is not None:
            if torch.is_tensor(data):
                data = data.cpu().numpy()
            original_shape = data.shape
            n_features_for_scaler = self.scaler.mean_.shape[0]

            if data.ndim > 1 and data.shape[-1] != n_features_for_scaler:
                 data_reshaped = data.reshape(-1, n_features_for_scaler)
                 return self.scaler.inverse_transform(data_reshaped).reshape(original_shape)
            else:
                 return self.scaler.inverse_transform(data)
        return data


# --- GRU Model Definition ---

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_features, seq_len, pred_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_features = output_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_size = input_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, pred_len * output_features)


    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.gru(x, h0) # GRU returns output and last hidden state

        # Take the output of the last time step: out[:, -1, :] shape: (batch_size, hidden_size)
        # Pass through the linear layer to get the flattened prediction
        flattened_prediction = self.fc(out[:, -1, :]) # Shape: (batch_size, pred_len * output_features)

        # Reshape the flattened prediction to (batch_size, pred_len, output_features)
        prediction = flattened_prediction.reshape(x.size(0), self.pred_len, self.output_features)

        return prediction # Shape: (batch_size, pred_len, output_features)

# --- Training and Evaluation Functions ---

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

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

    mse = MSE(preds, trues)
    mae = MAE(preds, trues)

    return mse, mae, total_loss # Return MSE, MAE, and average batch loss

# --- Main Execution ---

if __name__ == "__main__":
    all_results = {}

    # --- Hyperparameter Tuning for a specific PRED_LEN ---
    # Define the PRED_LEN for tuning
    TUNING_PRED_LEN = PRED_LENS_TO_TEST[0] # Tune for PRED_LEN = 96

    print(f"\n--- Starting Hyperparameter Tuning for Pred Len: {TUNING_PRED_LEN} ---")
    print(f"Running Mode: {RUN_MODE}")
    print(f"Sequence Length: {SEQ_LEN}")

    # Define search space for hyperparameters
    hidden_sizes_to_test = [32, 64, 100]
    learning_rates_to_test = [0.01, 0.001, 0.0001]
    tuning_epochs = 5 # Train for fewer epochs during tuning

    best_val_mse = float('inf')
    best_hyperparameters = None
    tuning_results = {}

    # --- Data Loading for Tuning ---
    # Load data once for the tuning PRED_LEN
    print("\nLoading and preprocessing data for tuning...")
    load_start_time = time.time()

    df_raw_check_size = pd.read_csv(os.path.join(ROOT_PATH, DATA_PATH))
    if FEATURES == 'M':
        cols_data_check_size = [col for col in df_raw_check_size.columns if col != 'date']
        INPUT_SIZE_TUNING = len(cols_data_check_size) # Use specific input size for tuning PRED_LEN
        OUTPUT_FEATURES_TUNING = INPUT_SIZE_TUNING
    elif FEATURES == 'S':
         if TARGET is None or TARGET not in df_raw_check_size.columns:
             raise ValueError(f"TARGET column '{TARGET}' not found for size check.")
         INPUT_SIZE_TUNING = 1
         OUTPUT_FEATURES_TUNING = 1
    else:
         raise ValueError(f"Unsupported features flag: {FEATURES}")

    temp_train_dataset_for_scaler = ElectricityDataset(
         root_path=ROOT_PATH, data_path=DATA_PATH, flag='train',
         size=[SEQ_LEN, 0, TUNING_PRED_LEN], features=FEATURES, target=TARGET,
         scale=True, timeenc=0, freq='h'
    )
    fitted_scaler = temp_train_dataset_for_scaler.scaler

    train_dataset_tune = ElectricityDataset(
        root_path=ROOT_PATH, data_path=DATA_PATH, flag='train',
        size=[SEQ_LEN, 0, TUNING_PRED_LEN], features=FEATURES, target=TARGET,
        scale=True, timeenc=0, freq='h', scaler=fitted_scaler
    )
    vali_dataset_tune = ElectricityDataset(
        root_path=ROOT_PATH, data_path=DATA_PATH, flag='val',
        size=[SEQ_LEN, 0, TUNING_PRED_LEN], features=FEATURES, target=TARGET,
        scale=True, timeenc=0, freq='h', scaler=fitted_scaler
    )

    train_loader_tune = DataLoader(train_dataset_tune, batch_size=32, shuffle=True)
    vali_loader_tune = DataLoader(vali_dataset_tune, batch_size=32, shuffle=False)

    load_end_time = time.time()
    print(f"Data loading and preprocessing time for tuning: {load_end_time - load_start_time:.4f}s")
    print(f"Input size (features): {INPUT_SIZE_TUNING}, Output features: {OUTPUT_FEATURES_TUNING}")
    print(f"Sequence length: {SEQ_LEN}, Tuning Prediction length: {TUNING_PRED_LEN}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Tuning Loops ---
    for hidden_size in hidden_sizes_to_test:
        for learning_rate in learning_rates_to_test:
            print(f"\n--- Trying Hyperparameters: Hidden Size = {hidden_size}, Learning Rate = {learning_rate} ---")

            # Instantiate model and optimizer for current hyperparameters
            model_tune = GRUModel(INPUT_SIZE_TUNING, hidden_size, 2, OUTPUT_FEATURES_TUNING, SEQ_LEN, TUNING_PRED_LEN).to(device) # Assuming 2 layers fixed for tuning
            criterion_tune = nn.MSELoss()
            optimizer_tune = torch.optim.Adam(model_tune.parameters(), lr=learning_rate)

            # Train for tuning epochs
            print(f"Training for {tuning_epochs} epochs...")
            for epoch in range(tuning_epochs):
                 epoch_train_loss = train_one_epoch(model_tune, train_loader_tune, criterion_tune, optimizer_tune, device)
                 print(f"  Epoch [{epoch+1}/{tuning_epochs}], Train Loss: {epoch_train_loss:.4f}")

            # Evaluate on validation set
            print("Evaluating on validation set...")
            val_mse, val_mae, val_loss = evaluate(model_tune, vali_loader_tune, device)
            print(f"  Validation Metrics: MSE = {val_mse:.4f}, MAE = {val_mae:.4f}")

            # Store tuning results
            tuning_results[(hidden_size, learning_rate)] = {'MSE': val_mse, 'MAE': val_mae}

            # Check if this is the best model so far
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_hyperparameters = {'HIDDEN_SIZE': hidden_size, 'LEARNING_RATE': learning_rate, 'NUM_LAYERS': 2} # NUM_LAYERS is fixed in this example

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best Validation MSE: {best_val_mse:.4f}")
    print("Best Hyperparameters found:", best_hyperparameters)
    print("Tuning Results Summary:")
    for hp, metrics in tuning_results.items():
        print(f"  HP: {hp}, Val MSE: {metrics['MSE']:.4f}, Val MAE: {metrics['MAE']:.4f}")

    # --- Use Best Hyperparameters for Full Training and Testing on all Pred Lens ---
    print("\n--- Starting Full Training and Testing with Best Hyperparameters ---")

    # Use the best hyperparameters found during tuning
    BEST_HIDDEN_SIZE = best_hyperparameters['HIDDEN_SIZE']
    BEST_LEARNING_RATE = best_hyperparameters['LEARNING_RATE']
    BEST_NUM_LAYERS = best_hyperparameters['NUM_LAYERS']

    all_results = {} 

    for current_pred_len in PRED_LENS_TO_TEST:
        PRED_LEN = current_pred_len
        DESCRIPTION = BASE_DESCRIPTION.format(PRED_LEN)

        print(f"\n--- Running Full Training for {DESCRIPTION} ---")

        # --- Data Loading for Full Run ---
        print("Loading and preprocessing data...")
        load_start_time = time.time()

        # Re-determine sizes for the current PRED_LEN
        df_raw_check_size = pd.read_csv(os.path.join(ROOT_PATH, DATA_PATH))
        if FEATURES == 'M':
            cols_data_check_size = [col for col in df_raw_check_size.columns if col != 'date']
            INPUT_SIZE_RUN = len(cols_data_check_size)
            OUTPUT_FEATURES_RUN = INPUT_SIZE_RUN
        elif FEATURES == 'S':
             if TARGET is None or TARGET not in df_raw_check_size.columns:
                 raise ValueError(f"TARGET column '{TARGET}' not found for size check.")
             INPUT_SIZE_RUN = 1
             OUTPUT_FEATURES_RUN = 1
        else:
             raise ValueError(f"Unsupported features flag: {FEATURES}")

        # Re-create the fitted_scaler for the current PRED_LEN to ensure strict consistency
        temp_train_dataset_for_scaler_run = ElectricityDataset(
             root_path=ROOT_PATH, data_path=DATA_PATH, flag='train',
             size=[SEQ_LEN, 0, PRED_LEN], features=FEATURES, target=TARGET,
             scale=True, timeenc=0, freq='h'
        )
        fitted_scaler_run = temp_train_dataset_for_scaler_run.scaler


        train_dataset = ElectricityDataset(
            root_path=ROOT_PATH, data_path=DATA_PATH, flag='train',
            size=[SEQ_LEN, 0, PRED_LEN], features=FEATURES, target=TARGET,
            scale=True, timeenc=0, freq='h', scaler=fitted_scaler_run
        )
        vali_dataset = ElectricityDataset( # Load vali for potential monitoring during full training
             root_path=ROOT_PATH, data_path=DATA_PATH, flag='val',
             size=[SEQ_LEN, 0, PRED_LEN], features=FEATURES, target=TARGET,
             scale=True, timeenc=0, freq='h', scaler=fitted_scaler_run
        )
        test_dataset = ElectricityDataset(
            root_path=ROOT_PATH, data_path=DATA_PATH, flag='test',
            size=[SEQ_LEN, 0, PRED_LEN], features=FEATURES, target=TARGET,
            scale=True, timeenc=0, freq='h', scaler=fitted_scaler_run
        )


        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        vali_loader = DataLoader(vali_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


        load_end_time = time.time()
        print(f"Data loading and preprocessing time: {load_end_time - load_start_time:.4f}s")
        print(f"Input size (features): {INPUT_SIZE_RUN}, Output features: {OUTPUT_FEATURES_RUN}")
        print(f"Sequence length: {SEQ_LEN}, Prediction length: {PRED_LEN}")
        print(f"Using Best HPs: Hidden Size = {BEST_HIDDEN_SIZE}, Learning Rate = {BEST_LEARNING_RATE}, Num Layers = {BEST_NUM_LAYERS}")


        # --- Model, Optimizer, Criterion (using best HPs) ---
        OUTPUT_SIZE_MODEL_RUN = PRED_LEN * OUTPUT_FEATURES_RUN


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate model with current PRED_LEN and BEST HPs
        model = GRUModel(INPUT_SIZE_RUN, BEST_HIDDEN_SIZE, BEST_NUM_LAYERS, OUTPUT_FEATURES_RUN, SEQ_LEN, PRED_LEN).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LEARNING_RATE)


        # --- Training (Full Epochs) ---
        NUM_EPOCHS_FULL_TRAIN = 50 # Adjust number of full training epochs
        print(f"\nTraining for {NUM_EPOCHS_FULL_TRAIN} epochs on the full training set...")
        for epoch in range(NUM_EPOCHS_FULL_TRAIN):
             epoch_start_time = time.time()
             train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
             epoch_end_time = time.time()
             print(f"Epoch [{epoch+1}/{NUM_EPOCHS_FULL_TRAIN}] finished. Train Loss: {train_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.4f}s")

             if vali_loader is not None and (epoch + 1) % 10 == 0: # Evaluate every 10 epochs
                  vali_mse, vali_mae, vali_loss_val = evaluate(model, vali_loader, device)
                  print(f"  Validation Metrics (Epoch {epoch+1}): MSE = {vali_mse:.4f}, MAE = {vali_mae:.4f}")
                  model.train()


        # --- Testing ---
        print("\nStarting testing...")
        test_mse, test_mae, test_loss = evaluate(model, test_loader, device)

        print("\n--- Finished training and testing for Pred Len:", PRED_LEN, "---")
        print("Final Test Metrics:")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")

        # Store results for this PRED_LEN
        all_results[PRED_LEN] = {'MSE': test_mse, 'MAE': test_mae}

    # --- Summarize Results ---
    print("\n--- Summary of Results for All Prediction Lengths (using best HPs) ---")
    print(f"Running Mode: {RUN_MODE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Best Hyperparameters found (tuned on Pred Len {TUNING_PRED_LEN}): {best_hyperparameters}")
    for pred_len, metrics in all_results.items():
        print(f"  Pred Len {pred_len}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")