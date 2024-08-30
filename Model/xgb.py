# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import wandb
import joblib
from wandb.xgboost import WandbCallback

train_df = pd.read_csv('~/zebraMD/data/train_data.csv')

# Prepare the data
X = train_df.drop('prognosis', axis=1)
y = train_df['prognosis']

# Step 1: Encode the string labels
le = joblib.load('zebraMD/Model/label_mapping.joblib')
y_encoded = y.map(le)

def objective():
    # Initialize a new wandb run
    run = wandb.init()
    
    # Get hyperparameters for this run
    config = wandb.config
    
    # Set up stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
    # Lists to store scores
    accuracies = []
    log_losses = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(le.keys()),
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'subsample': config.subsample,
            'colsample_bytree': config.colsample_bytree,
            'min_child_weight': config.min_child_weight,
            'gamma': config.gamma,
            'eval_metric': 'mlogloss',
        }
        
        # Create WandbCallback instance
        wandb_callback = WandbCallback()
        
        # Train the model
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=config.num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[wandb_callback]
        )
        
        # Make predictions
        y_pred = model.predict(dval)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred_labels)
        logloss = log_loss(y_val, y_pred)
        
        accuracies.append(accuracy)
        log_losses.append(logloss)
        
        # Log metrics for this fold
        wandb.log({f'accuracy_fold_{fold}': accuracy, f'logloss_fold_{fold}': logloss})
    
    # Calculate and log average metrics
    avg_accuracy = np.mean(accuracies)
    avg_logloss = np.mean(log_losses)
    wandb.log({'avg_accuracy': avg_accuracy, 'avg_logloss': avg_logloss})
    
    # Return average logloss (to be minimized)
    return avg_logloss

# Step 3: Set up wandb sweep
sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {'name': 'avg_logloss', 'goal': 'minimize'},
    'parameters': {
        'max_depth': {'min': 3, 'max': 10},
        'learning_rate': {'min': 0.01, 'max': 0.3},
        'subsample': {'min': 0.5, 'max': 1.0},
        'colsample_bytree': {'min': 0.5, 'max': 1.0},
        'min_child_weight': {'min': 1, 'max': 10},
        'gamma': {'min': 0, 'max': 5},
        'num_boost_round': {'min': 50, 'max': 500}
    }
}

# %%
# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="zebraMD-xgboost-hyperparam-tuning")

# Run the sweep
wandb.agent(sweep_id, function=objective, count=20)  
print("Hyperparameter tuning completed.")
# %%
# After tuning, you can get the best hyperparameters and train a final model
# sweep = api.sweep(f"spuduch/zebraMD-xgboost-hyperparam-tuning/{sweep_id}")
api = wandb.Api()
sweep = api.sweep('zebraMD-xgboost-hyperparam-tuning/qr1z5oi6')

# Get the best run from the sweep
best_run = sweep.best_run()
best_config = best_run.config
print(best_config.keys())

# Train final model with best hyperparameters (you may want to train on all data or use cross-validation)
best_params = {
    'objective': 'multi:softprob',
    'num_class': len(le.keys()),
    'max_depth': best_config.get('max_depth'),
    'learning_rate': best_config.get('learning_rate'),
    'subsample': best_config.get('subsample'),
    'colsample_bytree': best_config.get('colsample_bytree'),
    'min_child_weight': best_config.get('min_child_weight'),
    'gamma': best_config.get('gamma'),
    'eval_metric': 'mlogloss',
    'n_estimators': best_config.get('num_boost_round'),
}

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X, y_encoded)

print("Final model trained with best hyperparameters.")

# Save the model
joblib.dump(final_model, '/usr4/ugrad/spuduch/zebraMD/Model/xgboost_model.joblib')
print("Model saved as 'xgboost_model.joblib'")
