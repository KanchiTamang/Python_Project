import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Load the dataset
data_path = 'hotel_bookings.csv'  # This should work if in the same directory
# OR specify the full path
# data_path = 'C:/Users/97797/OneDrive/Desktop/first_project_week8/Python_Project/hotel_bookings.csv'

df = pd.read_csv(data_path)


# Data preprocessing
df.fillna(method='ffill', inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df.drop('is_canceled', axis=1)  # Replace with the correct target variable
y = df['is_canceled']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Create a Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
