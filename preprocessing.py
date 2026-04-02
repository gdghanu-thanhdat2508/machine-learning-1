import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


#load the dataset
df = pd.read_csv('startup_success_dataset.csv')

#cleaning the dataset
df.columns = df.columns.str.strip()

#Target Encoding (y)
# Convert 'outcome' into numbers 
label_encoder = LabelEncoder()
df['outcome'] = label_encoder.fit_transform(df['outcome'])

# Store the mapping for later reference
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Outcome Mapping: {mapping}")

# 4. Feature Selection
# Separate Features (X) and Target (y)
X = df.drop('outcome', axis=1)
y = df['outcome']

# 5. Categorical Encoding (One-Hot Encoding)
# Using get_dummies for 'investor_type', 'sector', and 'founder_background'
# We use drop_first=True to avoid Multicollinearity (important for Logistic Regression!)
X = pd.get_dummies(X, columns=['investor_type', 'sector', 'founder_background'], drop_first=True, dtype=int)

# 6. Train/Test Split
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create explicit copies to prevent pandas SettingWithCopyWarning when scaling
X_train = X_train.copy()
X_test = X_test.copy()

# 7. Feature Scaling
# Since variables like 'revenue_million' are much larger than 'funding_rounds',
# we must scale them so the model treats them fairly.
scaler = StandardScaler()

# Identify numerical columns to scale
num_cols = ['funding_rounds', 'founder_experience_years', 'team_size', 
            'market_size_billion', 'product_traction_users', 
            'burn_rate_million', 'revenue_million']

# Fit on training data only to avoid Data Leakage
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\nPreprocessing Complete")
print(X_train.head())

# 8. Save Preprocessed Data to CSV
train_processed = X_train.copy()
train_processed['outcome'] = y_train
train_processed.to_csv('train_preprocessed.csv', index=False)

test_processed = X_test.copy()
test_processed['outcome'] = y_test
test_processed.to_csv('test_preprocessed.csv', index=False)
print("Preprocessed files saved as 'train_preprocessed.csv' and 'test_preprocessed.csv'")