import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load the trained model
model = joblib.load(open(r'xgboost_model', 'rb'))

# Title of the app
st.title("Income Prediction App")
st.write("This app predicts whether a person earns more than $50K or not based on the given inputs.")

img = Image.open(r'C:\Users\R-admin\farsana\Machine_learning\project\pred.jpg')
st.image(img, width=600)

# Function to map user-friendly education names to education.num
def education_to_num(education):
    education_map = {
        "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
        "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
        "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15,
        "Doctorate": 16
    }
    return education_map.get(education, 0)

# Function to determine age group
def get_age_group(age):
    if 17 <= age <= 30:
        return 'Youth'
    elif 31 <= age <= 50:
        return 'Middle-aged'
    else:
        return 'Senior'

# Function to determine hours worked category
def get_hours_category(hours):
    if hours <= 30:
        return 'Part-Time'
    elif hours <= 40:
        return 'Full-Time'
    elif hours <= 60:
        return 'Over-Time'
    else:
        return 'Extreme'

st.header("Input Features")
# Input grid
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    age_group = get_age_group(age)  # Automatically determine age group
    education = st.selectbox("Education Level",
                             options=["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", 
                                      "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", 
                                      "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]
                            )
    education_num = education_to_num(education)
    hours_per_week = st.number_input("Hours Per Week", min_value=1, max_value=100, value=40)
    hours_category = get_hours_category(hours_per_week)  # Automatically determine hours category

with col3:
    marital_status = st.selectbox("Marital Status", options=['Married-spouse-present', 'No-partner', 'Absent', 'Divorced'])
    capital_gain = st.slider("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.slider("Capital Loss", min_value=0, max_value=4356, value=0)

with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])
    occupation = st.selectbox("Occupation", options=['Exec-managerial','Prof-specialty', 'Craft-repair', 'Other-service'])
    relationship = st.selectbox("Relationship", options=['Own-child','Other'])

# Encoding categorical inputs
age_group_map = {'Youth': 0, 'Middle-aged': 1, 'Senior': 2}
hours_category_map = {'Part-Time': 0, 'Full-Time': 1, 'Over-Time': 2, 'Extreme': 3}
marital_status_map = {'Married-spouse-present': 1, 'No-partner': 1, 'Absent': 0, 'Divorced': 0}
occupation_map = {'Exec-managerial': 1, 'Prof-specialty': 0, 'Craft-repair': 0, 'Other-service': 0}  
relationship_map = {'Own-child': 1, 'Other': 0} 
sex_map = {'Male': 1, 'Female': 0}

# Encoding
age_group_encoded = age_group_map[age_group]
hours_category_encoded = hours_category_map[hours_category]
marital_status_encoded = marital_status_map[marital_status]
occupation_encoded = occupation_map[occupation]
relationship_encoded = relationship_map[relationship]
sex_encoded = sex_map[sex]

# Calculate net capital activity (capital gain - capital loss)
net = capital_gain - capital_loss
capital_activity = 1 if (capital_gain > 0 or capital_loss > 0) else 0

# Combine all features into a single input array
input_features = [
    age, education_num, capital_gain, hours_per_week, age_group_encoded, net,
    hours_category_encoded, capital_activity, marital_status_encoded, 
    marital_status_map['No-partner'], occupation_encoded, relationship_encoded, sex_encoded
]

# Define selected features list
selected_features_list = ['age', 'education.num', 'capital.gain', 'hours.per.week', 'ageGroup', 'net', 
                          'hours_category', 'capital_activity', 'marital.status_Married-spouse-present',
                          'marital.status_No-partner', 'occupation_Exec-managerial', 'relationship_Own-child', 'sex_Male']

# Prediction button
if st.button("Predict"):
    # Convert input_features to a dataframe
    input_df = pd.DataFrame([input_features], columns=selected_features_list)
    # Make prediction
    prediction = model.predict(input_df)
    # Display prediction result
    if prediction[0] == 1:
        st.success("Prediction: The person earns more than $50K.")
    else:
        st.error("Prediction: The person earns $50K or less.")
