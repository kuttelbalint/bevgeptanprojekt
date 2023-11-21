import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # or median, mode

    def get_cleaned_df(self):
        # Read CSV
        df = pd.read_csv(self.file_path)

        # Data cleaning
        df_cleaned = df.drop_duplicates()

        # Handling missing values
        df_cleaned = self.imputer.fit_transform(df_cleaned)

        # Label encoding
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype == 'object':
                df_cleaned[column] = self.label_encoder.fit_transform(df_cleaned[column])

        # Feature Scaling
        df_cleaned = pd.DataFrame(self.scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

        return df_cleaned

    def balance_data(self, X, y):
        smote = SMOTE()
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced