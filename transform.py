import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



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
