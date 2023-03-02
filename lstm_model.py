import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from keras.models import load_model
import streamlit as st
from collections.abc import MutableMapping
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("C:/Users/ACIDIC/Desktop/ffproject/keras_main_model.h5")


# Define the scaler
scaler = MinMaxScaler()

# Define the function to predict stock prices
def predict(model, df):
    if type(df) != np.ndarray or df.size == 0:
         st.write("Input data is not a numpy array or is empty.")
         return None
    tickers=('TSLA','GOOG','AAPL','META','AMZN','NFLX','BIDU','SNAP','PINS','BABA','BTC-USD','ETH-USD','TSM','NVDA','ZOM')   
    df = df.reshape(df.shape[0], len(tickers), 1)
    df = scaler.transform(df)
    prediction = model.predict(df)

    return prediction

# Define the main function
def run():
    st.title("Stock price prediction app")
    start='2016-01-01'
    end='2022-12-31'

    st.title('Stock Price Prediction')
    tickers=('TSLA','GOOG','AAPL','META','AMZN','NFLX','BIDU','SNAP','PINS','BABA','BTC-USD','ETH-USD','TSM','NVDA','ZOM')
    dropdown=st.multiselect('Pick your Stock Ticker',tickers)
    df = yf.download(dropdown,start,end)
    df = df.astype(float)
    n_steps = 5
    n_features = 6
    df = df.values
    stock_data = np.zeros((df.shape[0] - n_steps + 1, n_steps, n_features))
    for i in range(stock_data.shape[0]):
        stock_data[i] = df[i:i+n_steps]    

    stock_data = np.array(df)
    stock_data = stock_data.reshape(stock_data.shape[0], 5, 15, 1)
    print(stock_data.dtype)
    print(stock_data.shape)

if st.button("Predict"):
        prediction = predict(model,df)
        st.write("Prediction: ", prediction)

# Run the app
if __name__ == "__main__":
    run()
