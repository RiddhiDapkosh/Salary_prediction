import streamlit as st
import pandas as pd
import pickle

# Load the trained model
# Make sure 'linear_regression_model.pkl' is in the same directory as app.py or provide the correct path
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Create input fields for features
# Note: For 'Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles',
#       you should ideally use the same LabelEncoder used during training.
#       For simplicity in this demo, we're accepting numerical inputs directly.
#       In a real application, you would load/re-create your LabelEncoders.

rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)
company_name = st.selectbox('Company Name', Company)
job_title = st.number_input('Job Title (Encoded)', min_value=0, value=200)
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5)
location = st.number_input('Location (Encoded)', min_value=0, value=10)
employment_status = st.number_input('Employment Status (Encoded)', min_value=0, value=1)
job_roles = st.number_input('Job Roles (Encoded)', min_value=0, value=0)


if st.button('Predict Salary'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': company_name,
        'Job Title': job_title,
        'Salaries Reported': salaries_reported,
        'Location': location,
        'Employment Status': employment_status,
        'Job Roles': job_roles
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: {prediction:,.2f} INR')
