import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('~/zebraMD/data/Symptoms_and_Diagnosis.csv')
df = df.drop(columns=['Unnamed: 133'])

# Shuffle the DataFrame - im pretty sure it wasn't shuffled originally
df_shuffled = df.sample(frac=1).reset_index(drop=True)

X = df_shuffled.drop('prognosis', axis=1)
y = df_shuffled['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # I'm stratifying by y so that the distribution of the target is the same in both the training and testing sets

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv('~/zebraMD/data/train_data.csv', index=False) # saving the data
test_df.to_csv('~/zebraMD/data/test_data.csv', index=False)

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# making sure i did the stratification right
print("\ndist in original data:")
print(df['prognosis'].value_counts(normalize=True))
print("\ndist in training data:")
print(train_df['prognosis'].value_counts(normalize=True))
print("\ndist in testing data:")
print(test_df['prognosis'].value_counts(normalize=True))