import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from keras.models import load_model
import streamlit as st
from collections.abc import MutableMapping
import model

start="2015-01-01"
end="2022-12-31"



st.title('Stock Price Prediction')
tickers = ['TSLA', 'GOOG', 'AAPL', 'META', 'AMZN', 'NFLX', 'BIDU', 'SNAP','BTC-USD']

dropdown=st.selectbox('Pick your Stock Ticker',tickers)

# Download the stock data for the selected ticker
df = yf.download(dropdown,start,end)

# Pre-process the data to fit the model
from model import df_to_windowed_df

windowed_df = df_to_windowed_df(df, '2021-03-25', '2022-03-23', n=3)

# Pass the pre-processed data to the model for prediction
predictions = model(windowed_df)

# Plot the predictions and the original data
from model import dates_train, y_train, dates_val, y_val, dates_test, y_test,dates
from model import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
st.subheader('Predictions')
fig=plt.figure(figsize=(12,6))
plt.plot(dates, predictions)
plt.plot(dates, y_test)
plt.legend(['Predictions', 'Observations'])
st.pyplot(fig)
