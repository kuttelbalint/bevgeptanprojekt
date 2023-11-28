import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes model
from sklearn.metrics import classification_report, confusion_matrix
from dataProcessor import DataProcessor  # Assuming DataProcessor is a custom module


processor = DataProcessor('mushroom.csv')
df = processor.get_cleaned_df()


X = df.drop('population', axis=1)  
y = df['population']

X_balanced, y_balanced = processor.balance_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
