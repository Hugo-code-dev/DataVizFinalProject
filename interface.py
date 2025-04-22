import pandas as pd
import numpy as np
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
df['log_price'] = np.log(df['price'])
df['price_per_sqft'] = df['price'] / df['sqft_living']
print("\nNew variables:")
print(df[['log_price', 'price_per_sqft']].describe())


# ==========================
# 5. Prediction with a Linear Model
# ==========================
features = ['grade', 'sqft_living', 'bathrooms', 'lat', 'bedrooms', 'view', 'waterfront', 'sqft_above']
X = df[features]
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_log = linreg.predict(X_test)
r2_log = r2_score(y_test, y_pred_log)
print("R² (log_price prediction):", r2_log)

mse_log = mean_squared_error(y_test, y_pred_log)
rmse_log = np.sqrt(mse_log)
print("RMSE (log_price):", rmse_log)

y_pred_price = np.exp(y_pred_log)
y_test_price = np.exp(y_test)
mse_price = mean_squared_error(y_test_price, y_pred_price)
rmse_price = np.sqrt(mse_price)
print("RMSE (USD):", rmse_price)

# ==========================
# 6. GUI for Predicting House Prices (Tkinter)
# ==========================
import tkinter as tk

def predict_price():
    try:
        grade_val = int(entry_grade.get())
        sqft_living_val = float(entry_sqft_living.get())
        bathrooms_val = float(entry_bathrooms.get())
        lat_val = float(entry_lat.get())
        bedrooms_val = int(entry_bedrooms.get())
        view_val = int(entry_view.get())
        waterfront_val = int(entry_waterfront.get())
        sqft_above_val = float(entry_sqft_above.get())
    except ValueError:
        result_label.config(text="Error : You must enter valid values.")
        return
    
    house_data = {
        'grade': [grade_val],
        'sqft_living': [sqft_living_val],
        'bathrooms': [bathrooms_val],
        'lat': [lat_val],
        'bedrooms': [bedrooms_val],
        'view': [view_val],
        'waterfront': [waterfront_val],
        'sqft_above': [sqft_above_val]
    }
    house_df = pd.DataFrame(house_data)
    
    predicted_log = linreg.predict(house_df)
    predicted_price_val = np.exp(predicted_log)[0]
    
    result_label.config(text=f"Price predicted : {predicted_price_val:,.2f} USD")

root = tk.Tk()
root.title("House Price Prediction")

label_grade = tk.Label(root, text="Grade :")
label_grade.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_grade = tk.Entry(root)
entry_grade.grid(row=0, column=1, padx=5, pady=5)
entry_grade.insert(0, "7")

label_sqft_living = tk.Label(root, text="Sqft Living :")
label_sqft_living.grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_sqft_living = tk.Entry(root)
entry_sqft_living.grid(row=1, column=1, padx=5, pady=5)
entry_sqft_living.insert(0, "2000")

label_bathrooms = tk.Label(root, text="Bathrooms :")
label_bathrooms.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entry_bathrooms = tk.Entry(root)
entry_bathrooms.grid(row=2, column=1, padx=5, pady=5)
entry_bathrooms.insert(0, "3")

label_lat = tk.Label(root, text="Latitude :")
label_lat.grid(row=3, column=0, padx=5, pady=5, sticky="e")
entry_lat = tk.Entry(root)
entry_lat.grid(row=3, column=1, padx=5, pady=5)
entry_lat.insert(0, "47.5112")

label_bedrooms = tk.Label(root, text="Bedrooms :")
label_bedrooms.grid(row=4, column=0, padx=5, pady=5, sticky="e")
entry_bedrooms = tk.Entry(root)
entry_bedrooms.grid(row=4, column=1, padx=5, pady=5)
entry_bedrooms.insert(0, "3")

label_view = tk.Label(root, text="View (0 à 4) :")
label_view.grid(row=5, column=0, padx=5, pady=5, sticky="e")
entry_view = tk.Entry(root)
entry_view.grid(row=5, column=1, padx=5, pady=5)
entry_view.insert(0, "0")

label_waterfront = tk.Label(root, text="Waterfront (0=No,1=Yes) :")
label_waterfront.grid(row=6, column=0, padx=5, pady=5, sticky="e")
entry_waterfront = tk.Entry(root)
entry_waterfront.grid(row=6, column=1, padx=5, pady=5)
entry_waterfront.insert(0, "0")

label_sqft_above = tk.Label(root, text="Sqft Above :")
label_sqft_above.grid(row=7, column=0, padx=5, pady=5, sticky="e")
entry_sqft_above = tk.Entry(root)
entry_sqft_above.grid(row=7, column=1, padx=5, pady=5)
entry_sqft_above.insert(0, "1800")

predict_button = tk.Button(root, text="Predict", command=predict_price)
predict_button.grid(row=8, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="Prix prédit : ")
result_label.grid(row=9, column=0, columnspan=2, pady=10)

root.mainloop()