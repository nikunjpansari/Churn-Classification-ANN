import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Helper function to load an external CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the external CSS file
local_css("style.css")

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Sidebar navigation for switching between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About"])

if page == "Prediction":
    st.title('Customer Churn Prediction')
    
    # Use two columns for side-by-side inputs
    col1, col2 = st.columns(2)
    
    # Wrap column inputs in a div for center alignment
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 35)
        balance = st.number_input('Balance', value=0.0, format="%.2f")
        credit_score = st.number_input('Credit Score', value=600)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        estimated_salary = st.number_input('Estimated Salary', value=50000.0, format="%.2f")
        tenure = st.slider('Tenure', 0, 10, 5)
        num_of_products = st.slider('Number of Products', 1, 4, 2)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active_member = st.selectbox('Is Active Member', [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    
    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Combine input data with the encoded geography data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Perform prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Display the results with enhanced feedback styling
    st.markdown("### Prediction Result")
    st.write(f'**Churn Probability:** {prediction_proba:.2f}')
    
    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
    
elif page == "About":
    st.title('About This Application')
    st.markdown("""
    **Customer Churn Prediction App**
    
    This application uses a pre-trained deep learning model to forecast customer churn based on various inputs such as age, credit score, balance, geography, etc.
    
    **Features:**
    - **Clean and Responsive UI:** Enhanced with external CSS for an improved look and feel.
    - **Interactive Navigation:** Use the sidebar to switch between the prediction page and an about page.
    - **Comprehensive Data Processing:** Input data undergoes encoding and scaling before being fed into the prediction model.
    
    **How It Works:**
    1. **Input Collection:** Users provide customer details (e.g., demographics, account info).
    2. **Data Preprocessing:** Data is preprocessed using label encoding, one-hot encoding, and scaling.
    3. **Model Prediction:** The processed data is passed to a neural network which returns a churn probability.
    4. **Result Display:** The probability is shown along with a clear indicator of whether churn is likely.
    
    **About the Developer:**
    This demo app was built using Python, Streamlit, TensorFlow, and scikit-learn to demonstrate how machine learning can be integrated into web applications for business insights.
    
    For more details, please refer to the documentation or contact the developer.
    """)
