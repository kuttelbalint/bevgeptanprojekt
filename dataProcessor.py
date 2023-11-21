import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='mean')  # For numeric columns
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')  # For categorical columns

    def balance_data(self, X, y):
        smote = SMOTE()
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced

    def get_cleaned_df(self):
        # Read CSV
        df = pd.read_csv(self.file_path)

        # Data cleaning
        df_cleaned = df.drop_duplicates()

        # Separate numeric and categorical columns
        numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

        print("Numeric Columns:", numeric_cols)
        print("Categorical Columns:", categorical_cols)

        # Check if there are numeric columns
        if len(numeric_cols) > 0:
            # Handling missing values for numeric columns
            df_cleaned[numeric_cols] = self.imputer_numeric.fit_transform(df_cleaned[numeric_cols])
        else:
            print("No numeric columns to impute.")

        # Check if there are categorical columns
        if len(categorical_cols) > 0:
            # Handling missing values for categorical columns
            df_cleaned[categorical_cols] = self.imputer_categorical.fit_transform(df_cleaned[categorical_cols])
        else:
            print("No categorical columns to impute.")

        # Label encoding for categorical columns
        for column in categorical_cols:
            df_cleaned[column] = self.label_encoder.fit_transform(df_cleaned[column])

        # Feature Scaling
        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = pd.DataFrame(self.scaler.fit_transform(df_cleaned[numeric_cols]), columns=numeric_cols)
        else:
            print("No numeric columns to scale.")

        return df_cleaned
