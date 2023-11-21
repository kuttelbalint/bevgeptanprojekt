from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from dataProcessor import DataProcessor


processor = DataProcessor('mushroom.csv')
df = processor.get_cleaned_df()


X = df.drop('population', axis=1)
y = df['population']


# Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = df.drop('population', axis=1)
y = df['population']

# Balancing the dataset
X_balanced, y_balanced = processor.balance_data(X, y)

# Now you can split the data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameters grid
param_grid = {
    'n_estimators': [10 ,100,150, 200, 250, 300, 350, 400],
    'max_depth': [20, 30, 40, 50],         
    'min_samples_split': [2, 3, 4],        
    'min_samples_leaf': [1, 2, 3],       
    'bootstrap': [True]  
}

# Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
print("Best parameters:", best_params)
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)