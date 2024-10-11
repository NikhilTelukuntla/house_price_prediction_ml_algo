import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title of the app
st.title('House Price Prediction')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')

def user_input_features():
    Area = st.sidebar.slider('Area (sq ft)', 500, 10000, 3000)
    Bedrooms = st.sidebar.slider('Number of Bedrooms', 1, 10, 3)
    Bathrooms = st.sidebar.slider('Number of Bathrooms', 1, 5, 2)
    Stories = st.sidebar.slider('Number of Stories', 1, 4, 2)
    Guestroom = st.sidebar.selectbox('Guestroom', ['yes', 'no'])
    Basement = st.sidebar.selectbox('Basement', ['yes', 'no'])
    Hotwaterheating = st.sidebar.selectbox('Hotwaterheating', ['yes', 'no'])
    Airconditioning = st.sidebar.selectbox('Airconditioning', ['yes', 'no'])
    Parking = st.sidebar.slider('Parking Spaces', 0, 4, 2)
    Prefarea = st.sidebar.selectbox('Prefarea', ['yes', 'no'])
    Furnishingstatus = st.sidebar.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

    data = {
        'area': Area,
        'bedrooms': Bedrooms,
        'bathrooms': Bathrooms,
        'stories': Stories,
        'guestroom': 1 if Guestroom == 'yes' else 0,
        'basement': 1 if Basement == 'yes' else 0,
        'hotwaterheating': 1 if Hotwaterheating == 'yes' else 0,
        'airconditioning': 1 if Airconditioning == 'yes' else 0,
        'parking': Parking,
        'prefarea': 1 if Prefarea == 'yes' else 0,
        # Convert 'furnishingstatus' using the same mapping as the training data
        'furnishingstatus': 1 if Furnishingstatus == 'furnished' else 0.5 if Furnishingstatus == 'semi-furnished' else 0
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Load and preprocess dataset
data = pd.read_csv('Housing.csv')

# Convert categorical variables to numerical
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 0.5, 'unfurnished': 0})

# Drop the 'mainroad' column if it's still present in the dataset
if 'mainroad' in data.columns:
    data = data.drop(columns=['mainroad'])

# Check for any remaining string columns in the dataset
non_numeric_columns = data.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    st.write(f"Non-numeric columns found in the dataset: {non_numeric_columns}")
    st.stop()

# Prepare features and target variable
X = data.drop(columns=['price'])
y = data['price']

# Train-test split and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make prediction
prediction = model.predict(input_df)

# Display prediction
st.write('## Predicted House Price:')
st.write(f'₹{prediction[0]:,.2f}')
