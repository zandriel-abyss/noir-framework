

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = 'datasource/processed/features_final_all_layers.csv'
OUTPUT_TRAIN = 'datasource/processed/train_features_phase2.csv'
OUTPUT_TEST = 'datasource/processed/test_features_phase2.csv'

# Load features
df = pd.read_csv(INPUT_FILE)

# Drop wallet_address if present
if 'wallet_address' in df.columns:
    df = df.drop(columns=['wallet_address'])

# Drop rows with missing target label
df = df.dropna(subset=['label'])

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Drop original label if needed
df = df.drop(columns=['label'])

# Separate features and target
X = df.drop(columns=['label_encoded'])
y = df['label_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Save to CSV
train_df = X_train.copy()
train_df['label'] = y_train
train_df.to_csv(OUTPUT_TRAIN, index=False)

test_df = X_test.copy()
test_df['label'] = y_test
test_df.to_csv(OUTPUT_TEST, index=False)

print(f"Saved training data to: {OUTPUT_TRAIN}")
print(f"Saved testing data to: {OUTPUT_TEST}")