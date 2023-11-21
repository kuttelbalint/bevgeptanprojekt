from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataProcessor import DataProcessor


processor = DataProcessor('mushroom.csv')
df = processor.get_cleaned_df()

# region Decision Tree


X = df.drop('population', axis=1)
y = df['population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

# Grid Search
"""
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50, 55, 60, 70, 80, 100, 150, 200, 1000],
    'min_samples_split': [1, 2, 3, 5, 6, 10, 15, 20, 100, 500, 1000],
    'min_samples_leaf': [1, 3, 2, 4, 5, 10, 15, 25, 50, 75, 100],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2', None],
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
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')


grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

best_clf = DecisionTreeClassifier(**best_params, random_state=42)
best_clf.fit(X_train, y_train)
y_pred_optimized = best_clf.predict(X_test)

y_pred_optimized = best_clf.predict(X_test)


print("Accuracy after optimization:", accuracy_score(y_test, y_pred_optimized))

cm = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# endregion
