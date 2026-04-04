import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer

def main():
    print("🚀 STEP 1: LOADING AND ANALYZING DATA...")
    try:
        df = pd.read_csv('startup_success_dataset.csv')
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Please check the file path.")
        return

    # =========================================================
    # STEP 2: LABEL TRANSFORMATION (TARGET COLUMN)
    # =========================================================
    success_states = ['IPO', 'Acquired', 'Success'] 
    
    def to_success_flag(value):
        value = str(value).strip()
        if value in success_states:
            return 1
        else:
            return 0

    df['is_success'] = df['outcome'].apply(to_success_flag)
    TARGET_COLUMN = 'is_success'

    # =========================================================
    # STEP 3: FEATURE MATRIX DEFINITION
    # =========================================================
    numeric_cols = [
        'funding_rounds', 'founder_experience_years', 'team_size', 
        'market_size_billion', 'product_traction_users', 
        'burn_rate_million', 'revenue_million'
    ]
    
    categorical_cols = [
        'investor_type', 'sector', 'founder_background' 
    ]

    # Validate dataset columns
    missing_cols = [] 
    for col in numeric_cols + categorical_cols:
        if col not in df.columns: 
            missing_cols.append(col) 
            
    if missing_cols:
        print(f"❌ Error: Your CSV is missing the following columns: {missing_cols}")
        return
    
    X = df[numeric_cols + categorical_cols].copy()
    y = df[TARGET_COLUMN] 

    # =========================================================
    # STEP 4: PREPROCESSING (IMPUTATION & ENCODING)
    # =========================================================
    print("🧹 STEP 2: CLEANING AND ENCODING DATA...")
    
    # Handle missing numeric values with Median
    num_imputer = SimpleImputer(strategy='median') 
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    
    # Handle missing categorical values with a constant 'Unknown'
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # One-Hot Encoding to avoid the Dummy Variable Trap
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # =========================================================
    # STEP 5: DATA SPLITTING (TRAIN - VALIDATION - TEST)
    # =========================================================
    print("✂️ STEP 3: SPLITTING AND SCALING DATA...")
    
    # Split 1: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # Split 2: Divide Temp into 15% Validation, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"   📦 Train Set:      {len(X_train)} samples")
    print(f"   📦 Validation Set: {len(X_val)} samples")
    print(f"   📦 Test Set:       {len(X_test)} samples")
    
    # Feature Scaling (Standardization)
    scaler = StandardScaler() 
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    # IMPORTANT: Only transform Validation and Test sets to prevent Data Leakage
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols]) 

    # =========================================================
    # STEP 6: MODEL TRAINING
    # =========================================================
    print("⚙️ STEP 4: TRAINING LOGISTIC REGRESSION MODEL...")
    
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # =========================================================
    # STEP 7: EVALUATION & PREDICTION
    # =========================================================
    
    # Evaluate on Validation Set
    y_pred_val = model.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    # Evaluate on Test Set
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1] # Probability of Success
    acc_test = accuracy_score(y_test, y_pred_test)

    print("\n" + "="*60)
    print("📊 MODEL EVALUATION REPORT")
    print("="*60)
    print(f"🎯 Validation Accuracy: {acc_val * 100:.2f}%")
    print(f"🏆 Test Accuracy:       {acc_test * 100:.2f}%\n")
    
    print("📋 Detailed Metrics on Test Set:")
    print(classification_report(y_test, y_pred_test))

    print("\n" + "="*60)
    print("🎯 DEMO: PREDICTED SUCCESS SCORE FOR TEST DATASET")
    print("="*60)
    
    for i in range(min(50, len(y_test))):
        actual_status = "Success (1)" if y_test.iloc[i] == 1 else "Failed (0)"
        predicted_prob = y_prob_test[i] * 100
        print(f"- Startup #{i+1:<2} | Actual: {actual_status:<12} | Predicted Probability: {predicted_prob:.2f}%")
        
    # =========================================================
    # STEP 8: DATA VISUALIZATION
    # =========================================================
    print("\n🎨 STEP 5: GENERATING VISUALIZATIONS...")
    
    plt.figure(figsize=(18, 5))

    # --- CHART 1: CONFUSION MATRIX ---
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Fail (0)', 'Predicted Success (1)'],
                yticklabels=['Actual Fail (0)', 'Actual Success (1)'])
    plt.title('1. Confusion Matrix', fontweight='bold')

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
    plt.title('2. ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")

    # --- CHART 3: FEATURE IMPORTANCE ---
    plt.subplot(1, 3, 3)
    feature_names = X_train.columns
    coefficients = model.coef_[0]
    
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefficients})
    feat_imp_df = feat_imp_df.sort_values(by='Weight', ascending=True)
    
    top_features = pd.concat([feat_imp_df.head(5), feat_imp_df.tail(5)])
    
    palette = []
    for x in top_features['Weight']:
        if x < 0:
            palette.append('red')
        else:
            palette.append('green')

    sns.barplot(
        x='Weight',
        y='Feature',
        hue='Feature',
        data=top_features,
        palette=palette,
        legend=False
    )
    plt.title('3. Top Feature Weights', fontweight='bold')
    plt.xlabel('Weight (Negative = Reduces Success, Positive = Drives Success)')
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()