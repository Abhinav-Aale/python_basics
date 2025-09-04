import os
print("Current Working Directory:", os.getcwd())
print("Files in directory:", os.listdir())
# Import necessary libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
data = pd.read_csv('automobile_data.csv')
# Drop string columns that can't be converted to numeric
data.drop(['Car_Name'], axis=1, inplace=True)  # or 'Bike_Name' if that's what your data has


# Data Cleaning & Preprocessing
# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Feature scaling
scaler = StandardScaler()
data[['Present_Price', 'Kms_Driven']] = scaler.fit_transform(data[['Present_Price', 'Kms_Driven']])

# Exploratory Data Analysis (EDA)
# Summary statistics
print(data.describe())

# Data visualization
sns.histplot(data['Selling_Price'], kde=True)
plt.title('Distribution of Selling Price')
plt.show()

sns.boxplot(x='Fuel_Type_Petrol', y='Selling_Price', data=data)
plt.title('Selling Price by Fuel Type')
plt.show()

# Correlation matrix
corr_matrix = data.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Model Training
# Split data into training and testing sets
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'{name} - RMSE: {rmse}, R2: {r2}, MAE: {mae}')\\\