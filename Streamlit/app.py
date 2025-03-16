import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from scikeras.wrappers import KerasClassifier




def load_model():
    return keras.models.load_model("csat_model.h5")


def load_scaler():
    scaler_path = 'D:\E-Commerce-Customer-Satisfaction/scaler.pkl'
    return joblib.load(scaler_path)

def load_feature_list():
    features_path = 'D:\E-Commerce-Customer-Satisfaction/features.pkl'
    return joblib.load(features_path)

def preprocess_new_data(data, features, numerical_features):
    empty_df = pd.DataFrame()
    for col in features:
        
        if col not in data.columns:
            empty_df[col] = 0
        elif col in numerical_features:
            empty_df[col] = data[col]
        else:
            empty_df[col] = 1
    return empty_df

st.set_page_config(page_title="CSAT Prediction APP")
st.header("Customer Satisfaction Prediction System")
st.subheader("Input Features for Prediction")

channel_name = st.selectbox("Channel Name", ["Email", "Inbound", "Outcall"])
category = st.selectbox("Category", ['Product Queries', 'Order Related', 'Returns', 'Cancellation', 'Shopzilla Related', 'Payments related', 'Refund Related', 'Feedback', 'Offers & Cashback', 'Onboarding related', 'Others', 'App/website'])
sub_category = st.selectbox("Sub-category", ['Life Insurance', 'Product Specific Information', 'Installation/demo', 'Reverse Pickup Enquiry', 'Not Needed', 'Fraudulent User', 'Exchange / Replacement', 'Missing', 'General Enquiry', 'Return request', 'Delayed', 'Service Centres Related', 'Payment related Queries', 'Order status enquiry', 'Return cancellation', 'Unable to track', 'Seller Cancelled Order', 'Wrong', 'Invoice request', 'Priority delivery', 'Refund Related Issues', 'Signup Issues', 'Online Payment Issues', 'Technician Visit', 'UnProfessional Behaviour', 'Damaged', 'Product related Issues', 'Refund Enquiry', 'Customer Requested Modifications', 'Instant discount', 'Card/EMI', 'Shopzila Premium Related', 'Account updation', 'COD Refund Details', 'Seller onboarding', 'Order Verification', 'Other Cashback', 'Call disconnected', 'Wallet related', 'PayLater related', 'Call back request', 'Other Account Related Issues', 'App/website Related', 'Affiliate Offers', 'Issues with Shopzilla App', 'Billing Related', 'Warranty related', 'Others', 'e-Gift Voucher', 'Shopzilla Rewards', 'Unable to Login', 'Non Order related', 'Service Center - Service Denial', 'Payment pending', 'Policy Related', 'Self-Help', 'Commission related'])
order_date_time = st.text_input("Order Date Time (YYYY-MM-DD HH:MM:SS)")
issue_reported_at = st.text_input("Issue Reported At (YYYY-MM-DD HH:MM:SS)")
issue_responded = st.text_input("Issue Responded (YYYY-MM-DD HH:MM:SS)")
customer_city = st.text_input("Customer City")
product_category = st.selectbox("Product Category", ['LifeStyle', 'Electronics', 'Mobile', 'Home Appliences', 'Furniture', 'Home', 'Books & General merchandise', 'GiftCard', 'Affiliates'])
item_price = st.number_input("Item Price", min_value=0.0, step=0.01)
connected_handling_time = st.number_input("Connected Handling Time (seconds)", min_value=0.0, step=0.01)
agent_name = st.text_input("Agent Name")
supervisor = st.selectbox("Supervisor", ['Mason Gupta', 'Dylan Kim', 'Jackson Park', 'Olivia Wang', 'Austin Johnson', 'Emma Park', 'Aiden Patel', 'Evelyn Kimura', 'Nathan Patel', 'Amelia Tanaka', 'Harper Wong', 'Zoe Yamamoto', 'Scarlett Chen', 'Sophia Sato', 'Wyatt Kim', 'Logan Lee', 'Mia Patel', 'William Park', 'Emily Yamashita', 'Madison Kim', 'Noah Patel', 'Oliver Nguyen', 'Elijah Yamaguchi', 'Layla Taniguchi', 'Isabella Wong', 'Carter Park', 'Jacob Sato', 'Ethan Tan', 'Mia Yamamoto', 'Brayden Wong', 'Ava Wong', 'Landon Tanaka', 'Lucas Singh', 'Charlotte Suzuki', 'Abigail Suzuki', 'Ethan Nakamura', 'Olivia Suzuki', 'Alexander Tanaka', 'Lily Chen', 'Sophia Chen'])
manager = st.selectbox("Manager", ['Jennifer Nguyen', 'Michael Lee', 'William Kim', 'John Smith', 'Olivia Tan', 'Emily Chen'])
tenure_bucket = st.selectbox("Tenure Bucket", ['On Job Training', '>90', '0-30', '31-60', '61-90'])
agent_shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Split', 'Afternoon', 'Night'])
survey_response_date = st.text_input("Survey Response Date (01-Aug-23)")

if st.button("Predict CSAT Score"):
    new_data = pd.DataFrame({
        'channel_name': [channel_name],
        'category': [category],
        'Sub-category': [sub_category],
        'order_date_time': [order_date_time],
        'Issue_reported at': [issue_reported_at],
        'issue_responded': [issue_responded],
        'Customer_City': [customer_city],
        'Product_category': [product_category],
        'Item_price': [item_price],
        'connected_handling_time': [connected_handling_time],
        'Agent_name': [agent_name],
        'Supervisor': [supervisor],
        'Manager': [manager],
        'Tenure Bucket': [tenure_bucket],
        'Agent Shift': [agent_shift],
        'Survey_response_Date': [survey_response_date]
    })

    new_data['Issue_reported at'] = pd.to_datetime(new_data['Issue_reported at'], format='%d/%m/%Y %H:%M')
    new_data['issue_responded'] = pd.to_datetime(new_data['issue_responded'], format='%d/%m/%Y %H:%M')
    new_data['Response_Time_seconds'] = (new_data['issue_responded'] - new_data['Issue_reported at']).dt.total_seconds()
    new_data['order_date_time'] = pd.to_datetime(new_data['order_date_time'], format='%d/%m/%Y %H:%M')
    new_data['day_number_order_date'] = new_data['order_date_time'].dt.day
    new_data['Survey_response_Date'] = pd.to_datetime(new_data['Survey_response_Date'], format='%d-%b-%y')
    new_data['day_number_response_date'] = new_data['Survey_response_Date'].dt.day
    new_data['weekday_num_response_date'] = new_data['Survey_response_Date'].dt.weekday + 1
    new_data = new_data.drop(columns=['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date'])
    
    def rename_column(df,numerical_col):
        y=""
        for cols in df.columns.to_list():
            if cols not in numerical_col:
                y=df[cols]
                df.rename(columns={cols: cols+"_"+y[0]}, inplace=True)
        return df

    numerical_features = ['Item_price', 'connected_handling_time', 'Response_Time_seconds',
       'day_number_order_date', 'day_number_response_date',
       'weekday_num_response_date']
    
    new_data=rename_column(new_data,numerical_features)
    
    scaler = load_scaler()
    
    sorted_features = load_feature_list()
    
    
    new_data1 = preprocess_new_data(new_data, sorted_features, numerical_features)
    new_data1[numerical_features] = scaler.transform(new_data1[numerical_features])
    
    X_test_array = new_data1.values.astype(np.float32)
    
    
    # Load the model
    keras_model = load_model()
    
    # Make predictions
    predictions = keras_model.predict(X_test_array)
    pred_classes = np.argmax(predictions, axis=1)
    
    st.write("Prediction Results")
    st.write(f"The Predicted Customer Satisfaction Score is {int(pred_classes)+1}")
    
    
    
    #print(keras_model.summary())
    
    
            
    st.write("All Predictions:")
    st.write(predictions)
    

    
   
    

    
   
