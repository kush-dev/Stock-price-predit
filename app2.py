import pandas as pd
import TSLA
import GOOGLE 
import AAPL
import numpy as np
import pandas_datareader as pdr
import META
import AMZN
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from keras.models import load_model
import streamlit as st
from collections.abc import MutableMapping


start="2018-01-01"
end="2023-01-10"
total_tasks = 100

progress_bar = st.progress(0)
for i in range(100):
    
  with st.spinner("Running model..."):     
        
        progress_bar.progress(0)
        
        
        
        st.title('Stock Price Prediction')
        ticker_to_file = {'TSLA': 'tsla.py', 'GOOG': 'google.py','AAPL': 'apple.py','META': 'meta.py','AMZN': 'amzn.py'}
        st.empty()
        selected_ticker = st.sidebar.selectbox('Pick your Stock Ticker', list(ticker_to_file.keys()),key='ticker1')
        df=yf.download(selected_ticker)
        progress_bar.progress(i+20)




      #Describing Data

        st.subheader('Stock Data From 2020 - 2022')
        st.write(df)

        #Visualizations
        st.subheader('Closing Price vs Time Chart')
        fig=plt.figure(figsize=(16,8))
        plt.xlabel('Time in Years')
        plt.ylabel('Price')
        plt.plot(df.Close)
        st.pyplot(fig)
        progress_bar.progress(i+25)
        

        st.subheader('Closing Price vs Time Chart with 100ma and 200ma')
        ma100=df.Close.rolling(100).mean()
        ma200=df.Close.rolling(200).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100)
        plt.plot(ma200)
        plt.plot(df.Close)
        st.pyplot(fig)
        progress_bar.progress(i+35)
        
        
        
        
        progress_bar.progress(i+40)
        
        progress_bar.progress(i+60)
        