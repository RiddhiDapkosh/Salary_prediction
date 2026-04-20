%%writefile app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# Make sure 'linear_regression_model.pkl' is in the same directory as app.py or provide the correct path
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the original dataset to get categorical values for encoding
# Assuming the CSV is available at this path relative to the Streamlit app or uploaded
try:
    original_df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
except FileNotFoundError:
    st.error("Error: 'Salary_Dataset_DataScienceLovers.csv' not found. Please ensure it's in the same directory as app.py.")
    st.stop()

# Initialize and fit LabelEncoders for categorical columns
# This needs to be done here to ensure consistent encoding with the trained model
label_encoders = {}
categorical_cols = ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']

for col in categorical_cols:
    if col in original_df.columns:
        le = LabelEncoder()
        # Handle potential NaNs before fitting
        original_df[col] = original_df[col].fillna(original_df[col].mode()[0])
        le.fit(original_df[col])
        label_encoders[col] = le
    else:
        st.warning(f"Column '{col}' not found in the original dataset.")

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Create input fields for features
rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)

# For 'Company Name', use st.selectbox with actual company names
if 'Company Name' in label_encoders:
    company_names_list = label_encoders['Company Name'].classes_.tolist()
    selected_company_name_str = st.selectbox('Company Name', company_names_list)
    company_name_encoded = label_encoders['Company Name'].transform([selected_company_name_str])[0]
else:
    company_name_encoded = st.number_input('Company Name (Encoded - Fallback)', min_value=0, value=0)
    st.warning("Using fallback for Company Name as LabelEncoder was not available.")

# For other encoded columns, provide dropdowns/selectboxes if possible, or numerical input
# For simplicity, we'll keep number_input for now for others, but ideal would be to use selectbox with encoded values

if 'Job Title' in label_encoders:
    job_titles_list = label_encoders['Job Title'].classes_.tolist()
    selected_job_title_str = st.selectbox('Job Title', job_titles_list, index=job_titles_list.index('Data Scientist') if 'Data Scientist' in job_titles_list else 0)
    job_title_encoded = label_encoders['Job Title'].transform([selected_job_title_str])[0]
else:
    job_title_encoded = st.number_input('Job Title (Encoded - Fallback)', min_value=0, value=200)


if 'Location' in label_encoders:
    locations_list = label_encoders['Location'].classes_.tolist()
    selected_location_str = st.selectbox('Location', locations_list, index=locations_list.index('Bangalore') if 'Bangalore' in locations_list else 0)
    location_encoded = label_encoders['Location'].transform([selected_location_str])[0]
else:
    location_encoded = st.number_input('Location (Encoded - Fallback)', min_value=0, value=10)


if 'Employment Status' in label_encoders:
    emp_status_list = label_encoders['Employment Status'].classes_.tolist()
    selected_emp_status_str = st.selectbox('Employment Status', emp_status_list)
    employment_status_encoded = label_encoders['Employment Status'].transform([selected_emp_status_str])[0]
else:
    employment_status_encoded = st.number_input('Employment Status (Encoded - Fallback)', min_value=0, value=1)


if 'Job Roles' in label_encoders:
    job_roles_list = label_encoders['Job Roles'].classes_.tolist()
    selected_job_role_str = st.selectbox('Job Roles', job_roles_list)
    job_roles_encoded = label_encoders['Job Roles'].transform([selected_job_role_str])[0]
else:
    job_roles_encoded = st.number_input('Job Roles (Encoded - Fallback)', min_value=0, value=0)

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5)


if st.button('Predict Salary'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': company_name_encoded,
        'Job Title': job_title_encoded,
        'Salaries Reported': salaries_reported,
        'Location': location_encoded,
        'Employment Status': employment_status_encoded,
        'Job Roles': job_roles_encoded
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: {prediction:,.2f} INR')
