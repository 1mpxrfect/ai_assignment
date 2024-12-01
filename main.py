# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
# Replace 'kaggle_car_data.csv' with the path to your dataset
df = pd.read_csv('synthetic_car_data.csv')

# Standardize column names to avoid case and whitespace issues
df.columns = df.columns.str.strip().str.lower()
print("Available columns:", df.columns)

# Ensure the 'price' column exists
if 'price' not in df.columns:
    raise ValueError("The target column 'price' is not found in the dataset. Please check your dataset.")

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Handle missing values
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Preprocess the data
X_imputed = preprocessor.fit_transform(X)

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X_imputed[:, len(numeric_features):])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed[:, :len(numeric_features)])

# Combine numerical and categorical features
X_processed = np.hstack((X_scaled, X_encoded))

# Visualize the target variable distribution
plt.hist(y, bins=50, edgecolor='black')
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Performance:")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"R² Score: {r2_lr:.2f}")

# Train ANN model
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                         solver='adam', max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)
y_pred_ann = ann_model.predict(X_test)

# Evaluate ANN model
mse_ann = mean_squared_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)

print("\nArtificial Neural Network Performance:")
print(f"Mean Squared Error: {mse_ann:.2f}")
print(f"R² Score: {r2_ann:.2f}")

# Results comparison
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Artificial Neural Network'],
    'Mean Squared Error': [mse_lr, mse_ann],
    'R² Score': [r2_lr, r2_ann]
})

print("\nModel Comparison:")
print(results)

# Feature importance for Linear Regression
coefficients = lr_model.coef_
important_features = pd.DataFrame({
    'Feature': np.concatenate([numeric_features, encoder.get_feature_names_out()]),
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 Important Features (Linear Regression):")
print(important_features.head(10))

# Display sample predictions
sample_predictions = pd.DataFrame({
    'Actual Price': y_test.reset_index(drop=True),
    'Predicted Price (LR)': y_pred_lr,
    'Predicted Price (ANN)': y_pred_ann
})

print("\nSample Predictions:")
print(sample_predictions.head())

plt.figure(figsize=(10, 8))
sns.barplot(data=important_features.head(10), x='Coefficient', y='Feature')
plt.title("Top 10 Important Features (Linear Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()