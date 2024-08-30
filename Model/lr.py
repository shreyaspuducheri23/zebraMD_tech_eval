 # %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
import joblib
import matplotlib.pyplot as plt

train_df = pd.read_csv('~/zebraMD/data/train_data.csv')

X = train_df.drop('prognosis', axis=1)
y = train_df['prognosis']

# i'm using the label encoder from the nb.py file to maintain consistency
le = joblib.load('/usr4/ugrad/spuduch/zebraMD/Model/label_mapping.joblib')
y_encoded = y.map(le)

C_values = np.logspace(-2, 15, 20)  # 50 values from 10^-2 to 10^15

skf = StratifiedKFold(n_splits=5, shuffle=True)

avg_losses = {C: [] for C in C_values}

for C in C_values:
    print(f"Training models with C={C:.6f}")
    
    fold_losses = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]
        
        model = OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear', C=C))
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_val)
        
        # Calculate log loss
        loss = log_loss(y_val, y_pred_proba)
        fold_losses.append(loss)
        
    # Calculate and store average loss for this C value
    avg_loss = np.mean(fold_losses)
    avg_losses[C].append(avg_loss)
    print(f"  Average Log Loss: {avg_loss:.4f}")

# Find the best C value
best_C = min(avg_losses, key=lambda C: avg_losses[C][0])
best_loss = avg_losses[best_C][0]

print(f"\nBest regularization parameter C: {best_C:.6f}")
print(f"Best average log loss: {best_loss:.4f}")
# %%
# Train final model with best C value
final_model = OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear', C=best_C))
final_model.fit(X, y_encoded)

print("Final model trained with best regularization parameter.")

# Save the model
joblib.dump(final_model, '/usr4/ugrad/spuduch/zebraMD/Model/l1_lr_model.joblib')
print("Model saved as 'l1_lr_model.joblib'")

# %%
# Plot the regulairzation path
plt.figure(figsize=(12, 6))
plt.semilogx(C_values, [avg_losses[C][0] for C in C_values])
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Average Log Loss')
plt.title('Regularization Path for L1 Logistic Regression')
plt.axvline(best_C, color='r', linestyle='--', label=f'Best C: {best_C:.6f}')
plt.legend()
plt.grid(True)
plt.savefig('/usr4/ugrad/spuduch/zebraMD/Model/regularization_path.png')
plt.close()

print("\nRegularization path plot saved as 'regularization_path.png'")

# %%
