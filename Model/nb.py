import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize
import joblib


train_df = pd.read_csv('~/zebraMD/data/train_data.csv')

X = train_df.drop('prognosis', axis=1)
y = train_df['prognosis']

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(classes)}")
print(f"Class distribution:\n{y.value_counts(normalize=True)}")# SHould be uniform
print("\n")

# Initialize the classifier
clf = BernoulliNB()

# setting up the 5-fold CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

accuracies = {label: [] for label in classes}
f1_scores = {label: [] for label in classes}
aupr_scores = {label: [] for label in classes}
auc_roc_scores = {label: [] for label in classes}
cross_entropy_losses = [] 

for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)
    print(f"Fold {fold}:")


    # Calculate metrics for each label
    for i, label in enumerate(classes):
        # For binary metrics, treat the current label as positive and all others as negative
        y_val_binary = (y_val == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        accuracy = accuracy_score(y_val_binary, y_pred_binary)
        f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)
        aupr = average_precision_score(y_val_binary, y_pred_proba[:, i])
        
        # Check if we have both classes present for AUC-ROC
        if len(np.unique(y_val_binary)) > 1:
            auc_roc = roc_auc_score(y_val_binary, y_pred_proba[:, i])
        else:
            auc_roc = np.nan

        accuracies[label].append(accuracy)
        f1_scores[label].append(f1)
        aupr_scores[label].append(aupr)
        auc_roc_scores[label].append(auc_roc)


        print(f"  Label: {label}")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1-score: {f1:.4f}")
        print(f"    AUPR: {aupr:.4f}")
        print(f"    AUC-ROC: {auc_roc:.4f}")

    cross_entropy = log_loss(y_val, y_pred_proba)
    cross_entropy_losses.append(cross_entropy)
    print(f"  Cross-Entropy Loss: {cross_entropy:.8f}")

    print()

# Calculate and print average metrics for each label
print("Average metrics across all folds for each label:")
for label in classes:
    print(f"Label: {label}")
    print(f"  Accuracy: {np.mean(accuracies[label]):.4f} (+/- {np.std(accuracies[label]):.4f})")
    print(f"  F1-score: {np.mean(f1_scores[label]):.4f} (+/- {np.std(f1_scores[label]):.4f})")
    print(f"  AUPR: {np.mean(aupr_scores[label]):.4f} (+/- {np.std(aupr_scores[label]):.4f})")
    print(f"  AUC-ROC: {np.nanmean(auc_roc_scores[label]):.4f} (+/- {np.nanstd(auc_roc_scores[label]):.4f})")
    print()

# Calculate and print average metrics across all labels
print("Average metrics across all labels:")
print(f"  Accuracy: {np.mean([np.mean(accuracies[label]) for label in classes]):.4f}")
print(f"  F1-score: {np.mean([np.mean(f1_scores[label]) for label in classes]):.4f}")
print(f"  AUPR: {np.mean([np.mean(aupr_scores[label]) for label in classes]):.4f}")
print(f"  AUC-ROC: {np.nanmean([np.nanmean(auc_roc_scores[label]) for label in classes]):.4f}")
print(f"  Cross-Entropy Loss: {np.mean(cross_entropy_losses):.4f} (+/- {np.std(cross_entropy_losses):.4f})")
# train the model on the entire dataset
clf.fit(X, y_encoded)

# save the model
joblib.dump(clf, 'zebraMD/Model/naive_bayes_model.pkl')

label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

# Save the mapping to a file
joblib.dump(label_mapping, 'zebraMD/Model/label_mapping.joblib')

print("Label Mapping:")
for label, code in label_mapping.items():
    print(f"{code}: {label}")


print("Model trained and saved successfully!")