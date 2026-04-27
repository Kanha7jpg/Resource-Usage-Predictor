import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class ResourceLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(ResourceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, sequence_length, hidden_size)
        
        # We only want the output from the last time step
        out = out[:, -1, :] 
        out = self.fc(out)
        
        # Add a dimension for predictability horizon (batch_size, 1, output_size)
        out = out.unsqueeze(1)
        return out


class AdvancedModelTrainer:
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = ResourceLSTM(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        logger.info("Training Advanced LSTM model...")
        
        # Convert numpy arrays to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Keep track of the best weights in memory (simplified)
                self.best_state = self.model.state_dict().copy()

        # Load best weights
        self.model.load_state_dict(self.best_state)
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_test_t).cpu().numpy()
            
        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds)**2))
        
        logger.info(f"Evaluating LSTM...")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        return mae, rmse, preds

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"LSTM model saved to {filepath}")

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        logger.info(f"LSTM model loaded from {filepath}")
        
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()
