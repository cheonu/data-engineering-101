import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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

processed_data = preprocessor.fit_transform(df)
print(processed_data)
