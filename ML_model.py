import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset from web
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

try:
    df = pd.read_csv(url, names=column_names)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Split features & target
X = df.iloc[:, :-1]  # First 8 columns
y = df.iloc[:, -1]   # Last column (Class: 0 or 1)

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101)

# Train Logistic Regression model
model = LogisticRegression(solver='liblinear')  # Fixes warnings
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Save the trained model
joblib.dump(model, 'save_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as well
print("Model saved successfully as 'save_model.pkl'!")
