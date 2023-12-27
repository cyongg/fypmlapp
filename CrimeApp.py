import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import streamlit_folium as st_folium
import folium


def main():
    with open('C:/Users/yongg/Downloads/Major_Crime_Indicators_Open_Data.csv', 'r') as file:
        for line in file:
            print(line)

    
    # Load the pre-trained Gradient Boosting model
    best_gb_model = joblib.load('best_gb_model.h5')
    a=0
    st.title("Crime Prediction App")

    # Get user input for crime features
    user_input = {}

    st.write("OCC = OCCURENCE")

    selected_columns = [
    'OCC_YEAR','OCC_MONTH','OCC_DAY','OCC_DOY','OCC_DOW','OCC_HOUR','DIVISION','PREMISES_TYPE','HOOD_158', 'UCR_EXT','LOCATION_TYPE'
    ]

    df_sample = pd.read_csv("C:/Major_Crime_Indicators_Open_Data.csv")
    df_sample = df_sample.dropna()
    label_encoder = LabelEncoder()
    # Example: Get user input using Streamlit text_input for all features used during training
    for column in selected_columns:
        if column == 'OCC_YEAR':
            date_value = st.date_input(f"Select Occur Date", min_value=datetime.date(2014, 1, 1),
            max_value=datetime.date(2023, 12, 31))
            user_input['OCC_YEAR'] = date_value.year if date_value else None
            user_input['OCC_MONTH'] = date_value.month if date_value else None
            user_input['OCC_DAY'] = date_value.day if date_value else None
            user_input['OCC_DOY'] = date_value.timetuple().tm_yday if date_value else None
            user_input['OCC_DOW'] = date_value.weekday().tm_yday if date_value else None
        elif column == 'OCC_MONTH':
            a = 1
        elif column == 'OCC_DAY':
            a = 1
        elif column == 'OCC_DOY':
            a = 1
        elif column == 'OCC_DOW':
            a = 1
        else:
            unique_values = [''] + sorted(df_sample[column].unique().tolist())
            user_input[column] = st.selectbox(f"Select {column}", unique_values)

    # Make a prediction when the user clicks the button
    if st.button("Predict"):
        input_dict = {key: [value] for key, value in user_input.items()}
        input_df = pd.DataFrame(input_dict)

    # Label encode categorical variables
    for column in input_df.select_dtypes(include='object').columns:
        input_df[column] = label_encoder.fit_transform(input_df[column])

    prediction = best_gb_model.predict(input_df)
    if prediction == 0:
        prediction = 'Assault'
    elif prediction == 1:
        prediction = 'Auto Theft'
    elif prediction == 2:
        prediction = 'Break And Enter'
    elif prediction == 3:
        prediction = 'Robbery'
    elif prediction == 4:
        prediction = 'Theft Over'
    st.success(f"Prediction: {prediction}")

    st.write("Remark :")
    st.write("HOOD_158 (House Street Numbers)")
    st.write("UCR_EXT (Codes of MCI_CATEGORY)")

if __name__ == "__main__":
    main()
