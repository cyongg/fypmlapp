# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming df_cleaned is your cleaned DataFrame
col_list = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOY', 'OCC_DOW', 'OCC_HOUR', 'MCI_CATEGORY', 'DIVISION',
            'PREMISES_TYPE', 'HOOD_158', 'UCR_EXT', 'LOCATION_TYPE']

df2 = df_cleaned[col_list]

# Assuming you have a DataFrame df with features X and target variable y
X = df2.drop(['MCI_CATEGORY'], axis=1)
y = df2['MCI_CATEGORY']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Label Encoding for target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize Gradient Boosting model
gb_model = GradientBoostingClassifier()

# Fit the model
gb_model.fit(X_train_encoded, y_train_encoded)

# Streamlit App
st.title("Crime Prediction App")

# Input form for user to input data
st.sidebar.header("User Input Features")
features = []
for col in X.columns:
    features.append(st.sidebar.slider(col, float(X[col].min()), float(X[col].max())))

# Reshape the input data for prediction
input_data = pd.DataFrame([features], columns=X.columns)
input_encoded = encoder.transform(input_data)
input_encoded = input_encoded.reshape(1, -1)  # Reshape for prediction

# Make predictions
prediction = gb_model.predict(input_encoded)
prediction_label = label_encoder.inverse_transform(prediction)[0]

st.header("Prediction")
st.write(f"The predicted crime category is: {prediction_label}")

# Display evaluation metrics on the test set
st.header("Model Evaluation on Test Set")
y_pred_test = gb_model.predict(X_test_encoded)
accuracy_test = accuracy_score(y_test_encoded, y_pred_test)
conf_matrix_test = confusion_matrix(y_test_encoded, y_pred_test)
class_report_test = classification_report(y_test_encoded, y_pred_test)

st.write(f"Accuracy on Test Set: {accuracy_test}")
st.write("Confusion Matrix on Test Set:")
st.write(conf_matrix_test)
st.write("Classification Report on Test Set:")
st.write(class_report_test)
