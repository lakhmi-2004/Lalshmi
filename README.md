# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Step 1: Load dataset
df = pd.read_csv("house_prices.csv")  # Replace with your dataset

# Step 2: Data preprocessing
df = df.dropna(subset=['SalePrice'])  # Drop rows with missing target

# Fill missing values for numerical and categorical
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna('None', inplace=True)

# Encode categorical variables
df = pd.get_dummies(df)

# Step 3: Feature and target split
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model - XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning with GridSearchCV
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(xgb_model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

# Step 7: Evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Best XGBoost Model: {grid.best_params_}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
