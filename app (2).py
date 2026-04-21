import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

st.title('💼 Salary Prediction App')
st.write('Enter the details below to predict salary.')

# -------------------------------
# Load Model
# -------------------------------
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# -------------------------------
# Load Dataset (FIXED PATH ✅)
# -------------------------------
try:
    original_df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
except FileNotFoundError:
    st.error("Dataset not found. Make sure CSV is in same folder as app.py")
    st.stop()

# -------------------------------
# Encoding
# -------------------------------
label_encoders = {}
categorical_cols = ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']

for col in categorical_cols:
    if col in original_df.columns:
        le = LabelEncoder()
        original_df[col] = original_df[col].fillna(original_df[col].mode()[0])
        le.fit(original_df[col])
        label_encoders[col] = le

# -------------------------------
# USER INPUT
# -------------------------------

rating = st.slider('Rating', 0.0, 5.0, 3.5)

# Company Name
company = st.selectbox('Company Name', label_encoders['Company Name'].classes_)
company_encoded = label_encoders['Company Name'].transform([company])[0]

# Job Title
job_title = st.selectbox('Job Title', label_encoders['Job Title'].classes_)
job_title_encoded = label_encoders['Job Title'].transform([job_title])[0]

# Location
location = st.selectbox('Location', label_encoders['Location'].classes_)
location_encoded = label_encoders['Location'].transform([location])[0]

# Employment Status
emp_status = st.selectbox('Employment Status', label_encoders['Employment Status'].classes_)
emp_status_encoded = label_encoders['Employment Status'].transform([emp_status])[0]

# Job Roles
job_role = st.selectbox('Job Roles', label_encoders['Job Roles'].classes_)
job_role_encoded = label_encoders['Job Roles'].transform([job_role])[0]

# Salaries Reported
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5)

# -------------------------------
# Prediction
# -------------------------------
if st.button('Predict Salary 💰'):

    # IMPORTANT: Column order MUST match training
    input_data = pd.DataFrame([[
        rating,
        company_encoded,
        job_title_encoded,
        salaries_reported,
        location_encoded,
        emp_status_encoded,
        job_role_encoded
    ]], columns=[
        'Rating',
        'Company Name',
        'Job Title',
        'Salaries Reported',
        'Location',
        'Employment Status',
        'Job Roles'
    ])

    prediction = model.predict(input_data)[0]

    st.success(f'💰 Predicted Salary: ₹ {prediction:,.2f}')
