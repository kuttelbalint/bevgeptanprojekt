import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='mean')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')

    def balance_data(self, X, y):
        smote = SMOTE()
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def advanced_impute(self, df):
        knn_imputer = KNNImputer(n_neighbors=5)
        return pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

    def one_hot_encode(self, df, categorical_cols):
        encoder = OneHotEncoder()
        encoded_df = pd.DataFrame(encoder.fit_transform(df[categorical_cols]).toarray())
        df = df.drop(categorical_cols, axis=1)
        df = df.join(encoded_df)
        return df
    
    def get_cleaned_df(self):
        df = pd.read_csv(self.file_path)
        
        df_cleaned = df.drop_duplicates()

        numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns


        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = self.imputer_numeric.fit_transform(df_cleaned[numeric_cols])
        else:
            print("No numeric columns to impute.")

        if len(categorical_cols) > 0:
            df_cleaned[categorical_cols] = self.imputer_categorical.fit_transform(df_cleaned[categorical_cols])
        else:
            print("No categorical columns to impute.")

        for column in categorical_cols:
            df_cleaned[column] = self.label_encoder.fit_transform(df_cleaned[column])

        # Feature Scaling
        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = pd.DataFrame(self.scaler.fit_transform(df_cleaned[numeric_cols]), columns=numeric_cols)
        else:
            print("No numeric columns to scale.")

        # Advanced imputation (optional)
        df_cleaned = self.advanced_impute(df_cleaned)

        # One-hot encoding (optional)
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        df_cleaned = self.one_hot_encode(df_cleaned, categorical_cols)

        return df_cleaned
