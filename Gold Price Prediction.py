import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

# Features: Date,Gold_Bees,Sensex,Nifty50,USD/EUR,Crude_Oil

# Load the dataset
gold_data = pd.read_csv('data_set.csv')

# Convert Crude_Oil and USD/EUR to INR
exchange_rate_inr_usd = 84.48
gold_data['Crude_Oil_INR'] = gold_data['Crude_Oil'] * exchange_rate_inr_usd
gold_data['EUR_USD_INR'] = gold_data['EUR/USD'] * exchange_rate_inr_usd

# Drop the original columns
gold_data = gold_data.drop(columns=['Crude_Oil', 'EUR/USD'])

# Convert the 'Date' column to datetime format
gold_data['Date'] = pd.to_datetime(gold_data['Date'], dayfirst=True)

# Calculate the correlation matrix and plot it
numeric_columns = gold_data.select_dtypes(include='number')
correlation = gold_data.corr()

plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

# Plot distribution of 'Gold_Bees'
sns.histplot(gold_data['Gold_Bees'], kde=True, color='green')
plt.title('Distribution of Gold_Bees')
plt.show()

# Prepare features (X) and target (Y)
X = gold_data.drop(['Date', 'Gold_Bees'], axis=1)
Y = gold_data['Gold_Bees']

# Split the data into training and test sets (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]                # Whether bootstrap samples are used when building trees
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=2),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2,
                           scoring='r2')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, Y_train)

# Output the best parameters and best score
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions using the best model
test_data_prediction = best_model.predict(X_test)

# Calculate R squared error of the predictions
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error with best model: ", error_score)

# Save the best model
joblib.dump(best_model, 'best_gold_price_prediction.pkl')

# Plot feature importance
plt.figure(figsize=(10, 6))
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title('Feature Importance')
plt.show()

# Plot learning curves to visualize overfitting
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, Y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score', color='blue')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score', color='red')
plt.title('Learning Curves')
plt.xlabel('Training Size')
plt.ylabel('Score (R^2)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Plot GridSearchCV performance
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
sns.heatmap(
    results.pivot_table(values='mean_test_score', index='param_max_depth', columns='param_n_estimators'),
    annot=True, fmt='.3f', cmap='viridis', cbar=True
)
plt.title('Grid Search Performance: Mean Test Scores for Different Hyperparameters')
plt.show()
