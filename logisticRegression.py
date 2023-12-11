from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from dataProcessor import DataProcessor
import seaborn as sns

# Data processing
processor = DataProcessor('mushroom.csv')
df = processor.get_cleaned_df()

X = df.drop('population', axis=1)
y = df['population']

X_balanced, y_balanced = processor.balance_data(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg_model = LogisticRegression(max_iter=5000)  # Increased max_iter

# Training the model
log_reg_model.fit(X_train_scaled, y_train)

# Predicting the Test set results
y_pred = log_reg_model.predict(X_test_scaled)
y_prob = log_reg_model.predict_proba(X_test_scaled)  # Probabilities for ROC-AUC

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')  # Corrected ROC-AUC calculation
cm = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Report
report = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "ROC-AUC Score": roc_auc,
    "Confusion Matrix": cm,
    "Classification Report": class_report
}

for key, value in report.items():
    print(f"{key}:\n{value}\n")

# Confusion matrix plot
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
