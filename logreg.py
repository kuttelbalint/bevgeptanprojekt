import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from dataProcessor import DataProcessor  # Import your DataProcessor class

# Load and preprocess the data
processor = DataProcessor('mushroom.csv')
df = processor.get_cleaned_df()

# Feature Selection: Select your features (X) and target variable (y)
# Exclude the target variable 'population' from the features
X = df.drop('population', axis=1)
y = df['population']

X_balanced, y_balanced = processor.balance_data(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Creating the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues

# Training the model
log_reg_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = log_reg_model.predict(X_test)

# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
