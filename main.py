from copyreg import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import joblib

# sklearn seaborn matplotlib joblib

def main():
    
    RANDOM_SEED = 42
    
    # LOAD DATA
    print("STEP 1: LOADING DATA...")
    try:
        df = pd.read_csv('startup_success_dataset.csv')
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Please check the file path.")
        return

    # DATA PREPARATION
    print("STEP 2: PREPARING DATA (TRANSFORMING LABELS, DEFINING FEATURES)...")
    success_states = ['IPO', 'Acquisition']
    
    def to_success_flag(value):
        value = str(value).strip()
        if value in success_states:
            return 1
        else:
            return 0

    df['is_success'] = df['outcome'].apply(to_success_flag)
    TARGET_COLUMN = 'is_success'

    #  Feature Matrix Definition 
    numeric_cols = [
        'funding_rounds', 'founder_experience_years', 'team_size', 
        'market_size_billion', 'product_traction_users',
        'burn_rate_million', 'revenue_million'
    ]
    
    categorical_cols = [
        'investor_type', 'sector', 'founder_background' 
    ]

    # Validate dataset columns
    missing_cols = [col for col in numeric_cols + categorical_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Your CSV is missing the following columns: {missing_cols}")
        return
    
    X = df[numeric_cols + categorical_cols].copy()
    y = df[TARGET_COLUMN]

    #  DATA SPLITTING 
    print("STEP 3: SPLITTING DATA INTO TRAIN, VALIDATION, AND TEST SETS...")
    
    # Split 1: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    # Split 2: Divide Temp into 15% Validation, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

    X_train = X_train.copy() # RAM
    X_val = X_val.copy()
    X_test = X_test.copy()

    print(f"Train Set:      {len(X_train)} samples")
    print(f"Validation Set: {len(X_val)} samples")
    print(f"Test Set:       {len(X_test)} samples")

    # STEP 4: PREPROCESSING (IMPUTATION, ENCODING & SCALING)
    print("🧹 STEP 4: PREPROCESSING DATA (IMPUTING, ENCODING, SCALING)...")
    
    #  Impute Numeric
    num_imputer = SimpleImputer(strategy='median') 
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = num_imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
    
    # Impute Categorical 
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = cat_imputer.transform(X_val[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # One-Hot Encoding - Prenvent Dummy Variable Trap
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
    X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Feature Scaling
    scaler = StandardScaler() 
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols]) 

    # STEP 5: MODEL TRAINING
    print("⚙️ STEP 5: TRAINING LOGISTIC REGRESSION MODEL...")
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED) # Classification
    model.fit(X_train, y_train)

    # MODEL EVALUATION
    print("STEP 6: EVALUATING MODEL PERFORMANCE...")
    #  Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1] # Success probabilities

    # Accuracy Scores
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    acc_test = accuracy_score(y_test, y_pred_test)

    print("\n" + "="*60)
    print("EVALUATION REPORT".center(60))
    print("="*60)
    print(f"Train Accuracy:      {acc_train * 100:.2f}%")
    print(f"Validation Accuracy: {acc_val * 100:.2f}%")
    print(f" Test Accuracy:       {acc_test * 100:.2f}%\n")
    
    print("📋 Detailed Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Failed (0)', 'Success (1)']))


    # VISUALIZATION
    print("\n🎨 STEP 7: GENERATING VISUALIZATIONS...")
    
    plt.figure(figsize=(18, 5))

    # --- CHART 1: CONFUSION MATRIX ---
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Fail (0)', 'Predicted Success (1)'],
                yticklabels=['Actual Fail (0)', 'Actual Success (1)'])
    plt.title('1. Confusion Matrix (Test Set)', fontweight='bold')

    # --- CHART 2: ROC CURVE ---
    plt.subplot(1, 3, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('2. ROC Curve (Test Set)', fontweight='bold')
    plt.legend(loc="lower right")

    # --- CHART 3: FEATURE IMPORTANCE ---
    plt.subplot(1, 3, 3)
    feature_names = X_train.columns
    coefficients = model.coef_[0]
    
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefficients})
    feat_imp_df = feat_imp_df.sort_values(by='Weight', ascending=True)
    
    top_features = pd.concat([feat_imp_df.head(5), feat_imp_df.tail(5)])
    
    # Create a color list based on weight
    top_features['Color'] = top_features['Weight'].apply(lambda x: 'green' if x > 0 else 'red')

    sns.barplot(
        x='Weight',
        y='Feature',
        data=top_features,
        palette=top_features['Color'].tolist() # Pass the exact color list
    )
    plt.title('3. Top Feature Weights', fontweight='bold')
    plt.xlabel('Weight (Negative = Reduces Success, Positive = Drives Success)')
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()
    
    print("\n💾 STEP 8: EXPORTING MODEL AND PREPROCESSING ARTIFACTS...")
    # Export the model
    export_bundle = {
        'model': model,
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'scaler': scaler,
        'expected_columns': X_train.columns.tolist()
    }
    
    export_path = 'startup_success_model.pkl'
    try:
        joblib.dump(export_bundle, export_path)
        print(f"✅ Model bundle successfully saved to '{export_path}'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
 
    
if __name__ == "__main__":
    main()