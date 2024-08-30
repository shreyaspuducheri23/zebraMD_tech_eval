# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import wandb
import joblib

train_df = pd.read_csv('~/zebraMD/data/train_data.csv')

# Prepare the data
X = train_df.drop('prognosis', axis=1)
y = train_df['prognosis']

le = joblib.load('/usr4/ugrad/spuduch/zebraMD/Model/saved_models/label_mapping.joblib')
y_encoded = y.map(le)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X.values).to('cuda' if torch.cuda.is_available() else 'cpu')
y_tensor = torch.LongTensor(y_encoded).to('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    def predict(self, x):
        self.eval()
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(x)
            # Get the predicted class (ind of the max value)
            _, predicted = torch.max(outputs, 1)
            return predicted
    def predict_proba(self, x):
        self.eval()
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(x)
            # Get the predicted class probabilities
            return nn.functional.softmax(outputs, dim=1)
        

# Training function
def train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        _, predicted = torch.max(val_outputs.data, 1)
        accuracy = accuracy_score(y_val.cpu().numpy(), predicted.cpu().numpy())
        val_proba = nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
        logloss = log_loss(y_val.cpu().numpy(), val_proba)
    
    return accuracy, logloss

# Objective function for wandb
def objective():
    # Initialize a new wandb run
    run = wandb.init()
    config = wandb.config
    
    # Set up stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    loglosses = []

    # copy tensors to CPU for sklearn compatibility
    X_cpu = X_tensor.cpu()
    y_cpu = y_tensor.cpu()

    for fold, (train_index, val_index) in enumerate(skf.split(X_cpu, y_cpu)):
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]
        
        model = SimpleNN(input_size=X.shape[1], hidden_size=config.hidden_size, num_classes=len(le.keys())).to('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        accuracy, logloss = train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, config.num_epochs)
        
        accuracies.append(accuracy)
        loglosses.append(logloss)
        
        wandb.log({f'accuracy_fold_{fold}': accuracy, f'logloss_fold_{fold}': logloss})
    
    avg_accuracy = np.mean(accuracies)
    avg_logloss = np.mean(loglosses)
    wandb.log({'avg_accuracy': avg_accuracy, 'avg_logloss': avg_logloss})
    
    return avg_logloss

# Set up wandb sweep
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'avg_logloss', 'goal': 'minimize'},
    'parameters': {
        'hidden_size': {'min': 32, 'max': 256},
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'num_epochs': {'min': 10, 'max': 100}
    }
}

# Initialize wandb

# Create the sweep
# sweep_id = wandb.sweep(sweep_config, project="zebraMD-nn-hyperparam-tuning")

# # Run the sweep
# wandb.agent(sweep_id, function=objective, count=20)  # Adjust count as needed

# print("Hyperparameter tuning completed.")

# # After tuning, you can get the best hyperparameters and train a final model
# api = wandb.Api()
# sweep = api.sweep(f'zebraMD-nn-hyperparam-tuning/l6epsgks')

# # Get the best run from the sweep
# best_run = sweep.best_run()
# best_config = best_run.config

# print(best_config.get('hidden_size'), best_config.get('learning_rate'), best_config.get('num_epochs'))
# # Train final model with best hyperparameters
# final_model = SimpleNN(input_size=X.shape[1], hidden_size=best_config.get('hidden_size'), num_classes=len(le.keys())).to('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(final_model.parameters(), lr=best_config.get('learning_rate'))

# # Train on all data
# final_model.train()
# for epoch in range(best_config.get('num_epochs')):
#     optimizer.zero_grad()
#     outputs = final_model(X_tensor)
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     optimizer.step()

# print("Final model trained with best hyperparameters.")

# # Save the model
# torch.save(final_model.state_dict(), '/usr4/ugrad/spuduch/zebraMD/Model/nn_model.pth')
# print("Model saved as 'nn_model.pth'")