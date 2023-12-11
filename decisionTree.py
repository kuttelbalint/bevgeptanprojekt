from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataProcessor import DataProcessor


processor = DataProcessor('mushroom.csv')
df = processor.get_cleaned_df()

# region Decision Tree

X = df.drop('population', axis=1)
y = df['population']
X_balanced, y_balanced = processor.balance_data(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

# Grid Search

param_grid = {
    'max_depth': [10, 20, 30, 40, 100, 250],  # More focused range
    'min_samples_split': [2, 10, 20, 50, 100],  # Start from 2
    'min_samples_leaf': [1, 5, 10, 20],  # Start from 1
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],  # Removed None
    'splitter': ['best', 'random']
}


"""
param_grid = {
    'max_depth': [None],
    'min_samples_split': [500, 200],
    'min_samples_leaf': [75, 60],
    'criterion': ['gini'],
    'max_features': [None],
    'splitter': ['random']
}"""

""" grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=10, n_jobs=-1, verbose=2, scoring='accuracy')


grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_ """
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, 
                                   n_iter=500, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_

best_clf = DecisionTreeClassifier(**best_params, random_state=42)
best_clf.fit(X_train, y_train)
y_pred_optimized = best_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred_optimized)
precision = precision_score(y_test, y_pred_optimized, average='weighted')
recall = recall_score(y_test, y_pred_optimized, average='weighted')
f1 = f1_score(y_test, y_pred_optimized, average='weighted')
cm = confusion_matrix(y_test, y_pred_optimized)
class_report = classification_report(y_test, y_pred_optimized)


report = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "ROC-AUC Score": roc_auc_score,
    "Confusion Matrix": cm,
    "Classification Report": class_report
}

for key, value in report.items():
    print(f"{key}:\n{value}\n")


sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# endregion
