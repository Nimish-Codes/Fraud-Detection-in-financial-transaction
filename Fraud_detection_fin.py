# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Hypothetical imbalanced dataset (replace this with your own dataset)
data = {
    'Amount': [100, 200, 150, 300, 500, 50, 80, 120, 200, 400],
    'Frequency': [5, 10, 7, 8, 20, 3, 4, 6, 15, 12],
    'IsFraud': [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]  # 1 represents fraud, 0 represents normal
}

df = pd.DataFrame(data)

# Streamlit App
st.title('Fraud Detection Model Evaluation')
st.write('This app evaluates a hypothetical fraud detection model.')

# Display sample data
st.subheader('Sample Data:')
st.write(df)

# Input fields for user
st.sidebar.title('User Input:')
amount_input = st.sidebar.number_input('Enter Amount:', min_value=0, max_value=1000, step=1, value=200)
frequency_input = st.sidebar.number_input('Enter Frequency:', min_value=0, max_value=20, step=1, value=10)

# Update the model with the user input
new_data = {'Amount': [amount_input], 'Frequency': [frequency_input]}
new_df = pd.DataFrame(new_data)

# Split the data into features (X) and labels (y)
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model with class_weight='balanced'
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
st.subheader('Model Evaluation:')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))
st.write('Classification Report:')
st.code(classification_report(y_test, y_pred, zero_division=1))  # Handle zero_division warning

# Example of using the model for prediction
prediction = model.predict(new_df)

st.subheader('Prediction for User Input:')
# st.write(f'Predicted class for user input: {prediction}')

if prediction == 0:
    st.success('No fraud detected')
else:
    st.warning('Warning! Fraud detected')
