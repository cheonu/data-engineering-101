import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt



# read csv
# def read_csv(csv_path):
#     df = pd.read_csv(csv_path)
#     return(df.to_string())
 
# df = read_csv("models/Titanic-Dataset.csv")

df = pd.read_csv("models/Titanic-Dataset.csv")

# Define transformations
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']  
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_cols = ['Sex', 'Embarked']    
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

X_transformed = preprocessor.fit_transform(df)
y = df['Survived']

# 1. Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test, = train_test_split (
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Learn the mapping f(x) → y
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Apply mapping to unseen data
y_pred = model.predict(X_test)

# 4. Evaluate generalization
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get human-readable names for your 10 dimensions
feature_names = preprocessor.get_feature_names_out()

# Extract learned weights (coefficients) from logistic regression
# model.coef_ shape: (1, 10) → flatten to (10,)
weights = model.coef_[0]

# Create interpretable dataframe
coef_df = pd.DataFrame({
    'feature': feature_names,
    'weight': weights,
    'abs_weight': np.abs(weights)
}).sort_values('abs_weight', ascending=False)

print("=== Feature Importance (Logistic Regression Weights) ===")
print(coef_df[['feature', 'weight']].to_string(index=False))


# Get probabilities for the test set
y_probs = model.predict_proba(X_test)[:,1]

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.', label='Model')
plt.xlabel('Recall (Catch true survivors)')
plt.ylabel('Precision (When we predict survive, how often right)')
plt.title('Precision-Recall Tradeoff')
plt.grid(True)
plt.legend()
plt.show()

# Find threshold that maximizes F1 (balance)
f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
best_idx = np.argmax(f1)
print(f"\nOptimal threshold (max F1): {thresholds[best_idx]:.3f}")
print(f"→ Precision: {precision[best_idx]:.3f}, Recall: {recall[best_idx]:.3f}")

# Show what happens at different thresholds
for thresh in [0.3, 0.5, 0.7]:
    preds = (y_probs >= thresh).astype(int)
    from sklearn.metrics import classification_report
    print(f"\n=== Threshold = {thresh} ===")
    print(classification_report(y_test, preds, target_names=['Died', 'Survived']))
