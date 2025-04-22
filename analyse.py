import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# 1. Data Loading and Pre-Filtering
# ==========================
df = pd.read_csv('kc_house_data.csv')
print("Initial dimensions:", df.shape)

# Filter outliers on 'price' using the IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print("Minimum price after IQR filter:", df['price'].min())
print("Maximum price after IQR filter:", df['price'].max())

initial_count = 21613
final_count = df.shape[0]
print(f"{initial_count - final_count} outliers removed. Intermediate dimensions: {df.shape}")

# Drop columns that are less relevant for correlation
cols_to_drop = [
    'id', 'date', 'sqft_lot', 'condition', 'sqft_lot15', 
    'sqft_living15', 'long', 'zipcode', 'yr_renovated', 
    'yr_built', 'sqft_basement'
]
df.drop(columns=cols_to_drop, inplace=True)
print("Remaining columns:", df.columns.tolist())

# Exclude houses with more than 10 bedrooms
df = df[df['bedrooms'] <= 10]
print("Final dimensions after excluding houses with more than 10 bedrooms:", df.shape)

# ==========================
# 2. Creating New Variables (including log transform)
# ==========================
# Apply log transform to 'price' to reduce skewness
df['log_price'] = np.log(df['price'])

# Create an additional variable (price_per_sqft)
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Display descriptive statistics for the new variables
print("\nNew variables:")
print(df[['log_price', 'price_per_sqft']].describe())

# ==========================
# 3. Previous Plots (EDA)
# ==========================

# 3.1. Correlation matrix (selected columns)
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
)
plt.title("Correlation Matrix (Selected Columns)", fontsize=14, pad=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 3.2. Boxplot of price (excluding outliers)
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['price'], color='skyblue')
plt.xlabel('Sale Price (USD)', fontsize=12)
plt.title('Boxplot of Price (Outliers Excluded)', fontsize=14, pad=10)
plt.show()

# 3.3. Histogram of the price distribution
plt.figure(figsize=(8, 5))
plt.hist(df['price'], bins=30, color='steelblue', edgecolor='black')
plt.xlabel('Sale Price (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title("Distribution of Price (Outliers Excluded)", fontsize=14, pad=10)
plt.tight_layout()
plt.show()

# 3.4. Bar chart: Average price by number of bedrooms
plt.figure(figsize=(8, 5))
price_mean_bed = df.groupby('bedrooms')['price'].mean().sort_index()
bars = plt.bar(price_mean_bed.index, price_mean_bed.values, color='cornflowerblue')
plt.xlabel("Number of Bedrooms", fontsize=12)
plt.ylabel("Average Price (USD)", fontsize=12)
plt.title("Average Price by Number of Bedrooms (Outliers Excluded)", fontsize=14, pad=10)
for i, bar in enumerate(bars):
    plt.text(
        bar.get_x() + bar.get_width()/2, bar.get_height(),
        f"{bar.get_height():.0f}", ha='center', va='bottom', 
        rotation=90, fontsize=8
    )
plt.tight_layout()
plt.show()

# ==========================
# 4. Additional Plots for the Log-Transformed Variable
# ==========================

# 4.1. Histogram of log_price
plt.figure(figsize=(6, 4))
sns.histplot(df['log_price'], kde=True, color='green', bins=30)
plt.title("Distribution of log_price", fontsize=14, pad=10)
plt.xlabel("log(Price)", fontsize=12)
plt.tight_layout()
plt.show()

# 4.2. Display the correlation of log_price with other variables
corr_log_price = df.corr(numeric_only=True)['log_price'].sort_values(ascending=False)
print("\nCorrelation with log_price:\n", corr_log_price)

# ==============================================
# 5. Prediction with a Linear Model
# ===============================================
# Select the features most correlated with log_price
features = ['grade', 'sqft_living', 'bathrooms', 'lat', 'bedrooms', 'view', 'waterfront', 'sqft_above']

X = df[features]
y = df['log_price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# Instantiate the model
linreg = LinearRegression()

# Train the model
linreg.fit(X_train, y_train)

# Retrieve coefficients
coeffs = pd.DataFrame({
    'Features': X_train.columns,
    'Coefficients': linreg.coef_
})
print("\nModel Coefficients:\n", coeffs)
print("\nIntercept (b):", linreg.intercept_)

# Predict on the test set
y_pred_log = linreg.predict(X_test)

# R²
r2_log = r2_score(y_test, y_pred_log)
print("R² (log_price prediction):", r2_log)

# MSE and RMSE
mse_log = mean_squared_error(y_test, y_pred_log)
rmse_log = np.sqrt(mse_log)
print("RMSE (log_price):", rmse_log)

# Convert predictions back to real price scale
y_pred_price = np.exp(y_pred_log)
y_test_price = np.exp(y_test)

mse_price = mean_squared_error(y_test_price, y_pred_price)
rmse_price = np.sqrt(mse_price)
print("RMSE (USD):", rmse_price)

# Plot: Real Price (USD) vs Predicted Price (USD)
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred_log)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_exp, y_pred_exp, alpha=0.5, color='teal', edgecolor='k')
# Reference diagonal
plt.plot(
    [y_test_exp.min(), y_test_exp.max()],
    [y_test_exp.min(), y_test_exp.max()],
    'r--', lw=2
)
plt.xlabel("Actual Price (USD)", fontsize=12)
plt.ylabel("Predicted Price (USD)", fontsize=12)
plt.title("Comparison of Actual vs. Predicted Price (USD)", fontsize=14, pad=12)
plt.tight_layout()
plt.show()

# Plot: log(Price) real vs. log(Price) predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_log, alpha=0.5, color='teal', edgecolor='k')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
plt.xlabel("log(Price) (Actual)", fontsize=12)
plt.ylabel("log(Price) (Predicted)", fontsize=12)
plt.title("Comparison of Actual log(Price) vs. Predicted log(Price)", fontsize=14, pad=12)
plt.tight_layout()
plt.show()

# ==========================
# 6. Example: Predicting Price for Custom House Parameters
# ==========================
# This is an example of how to pass custom parameters to the trained model.
# You can modify the values below to see how the predicted price changes.

house_data = {
    'grade': 7,
    'sqft_living': 2000,
    'bathrooms': 3,
    'lat': 47.5112,
    'bedrooms': 3,
    'view': 0,
    'waterfront': 1,
    'sqft_above': 1800
}

# Create a single-row DataFrame
house_df = pd.DataFrame([house_data])
predicted_log = linreg.predict(house_df)
predicted_price = np.exp(predicted_log)
print("\nCustom House Parameters:", house_data)
print(f"Predicted log(Price): {predicted_log[0]:.4f}")
print(f"Predicted Price (USD): {predicted_price[0]:.2f}")