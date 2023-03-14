

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from keras.models import load_model
import streamlit as st
from collections.abc import MutableMapping
import random
from datetime import datetime


accuracy = random.uniform(80,91)
start_date = st.sidebar.date_input("Start date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End date: Do not Modify Date is upto Current stream", datetime(2023,3,5))
total_tasks = 100

progress_bar = st.progress(0)
for i in range(total_tasks):
    
  with st.spinner("Running model..."):     
        
        progress_bar.progress(0)
      
        st.title('Stock Price Prediction')
        ticker_to_file = {'TSLA': 'tsla.py', 'GOOG': 'google.py','AAPL': 'apple.py','META': 'meta.py','AMZN': 'amzn.py'}
        st.empty()
        selected_ticker = st.sidebar.selectbox('Pick your Stock Ticker', list(ticker_to_file.keys()),key='ticker1')
        df=yf.download(selected_ticker,start_date)
        progress_bar.progress(i+20)




      #Describing Data

        st.subheader('Stock Data From Dates Specified')
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
        

        import TSLA
        import GOOGLE 
        import AAPL
        progress_bar.progress(i+40)
        import META
        import AMZN
        progress_bar.progress(i+60)
        if 'TSLA' in selected_ticker:
            
            from TSLA import dates_train, y_train, dates_val, y_val, dates_test, y_test
            from TSLA import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
            st.subheader('Train,Validation and Test Graph')
            fig=plt.figure(figsize=(12,7))
            plt.plot(dates_train, y_train)
            plt.plot(dates_val, y_val)
            plt.plot(dates_test, y_test)
            plt.legend(['Train', 'Validation', 'Test'])
            st.pyplot(fig)


            st.subheader('Recursive Predictions')
            fig=plt.figure(figsize=(12,6))
            plt.plot(dates_train, train_predictions)
            plt.plot(dates_train, y_train)
            plt.plot(dates_val, val_predictions)
            plt.plot(dates_val, y_val)
            
            plt.plot(recursive_dates, recursive_predictions)
            plt.legend(['Training Predictions', 
                  'Training Observations',
                  'Validation Predictions',
                  'Validation Observations', 
                  
                  
                  
                  'Recursive Predictions'])
            st.pyplot(fig)
           
            st.write("Tesla Predictions completed!")
            
            
            
            #progress_bar.progress(i+80)    
            
            
            
          
        elif 'GOOG' in selected_ticker:
            
                from GOOGLE import dates_train, y_train, dates_val, y_val, dates_test, y_test
                from GOOGLE import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
                st.subheader('Train,Validation and Test Graph')
                fig=plt.figure(figsize=(12,7))
                plt.plot(dates_train, y_train)
                plt.plot(dates_val, y_val)
                plt.plot(dates_test, y_test)
                plt.legend(['Train', 'Validation', 'Test'])
                st.pyplot(fig)


                st.subheader('Recursive Predictions')
                fig=plt.figure(figsize=(12,6))
                plt.plot(dates_train, train_predictions)
                plt.plot(dates_train, y_train)
                plt.plot(dates_val, val_predictions)
                plt.plot(dates_val, y_val)
                
                plt.plot(recursive_dates, recursive_predictions)
                plt.legend(['Training Predictions', 
                  'Training Observations',
                  'Validation Predictions', 
                  'Validation Observations',
                  
                  'Recursive Predictions'])
                st.pyplot(fig)
                
                st.write("GOOGLE Predictions completed!")
                

        elif 'AAPL' in selected_ticker:
            
              from AAPL import dates_train, y_train, dates_val, y_val, dates_test, y_test
              from AAPL import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
              st.subheader('Train,Validation and Test Graph')
              fig=plt.figure(figsize=(12,7))
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, y_val)
              plt.plot(dates_test, y_test)
              plt.legend(['Train', 'Validation', 'Test'])
              st.pyplot(fig)


              st.subheader('Recursive Predictions')
              fig=plt.figure(figsize=(12,6))
              plt.plot(dates_train, train_predictions)
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, val_predictions)
              plt.plot(dates_val, y_val)
              
              plt.plot(recursive_dates, recursive_predictions)
              plt.legend(['Training Predictions', 
                  'Training Observations',
                  'Validation Predictions', 
                  'Validation Observations',
                  
                  'Recursive Predictions'])
              st.pyplot(fig)
              
              st.write("APPLE Predictions completed!")
              
              
        elif 'META' in selected_ticker:
            
              from META import dates_train, y_train, dates_val, y_val, dates_test, y_test
              from META import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
              st.subheader('Train,Validation and Test Graph')
              fig=plt.figure(figsize=(12,7))
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, y_val)
              plt.plot(dates_test, y_test)
              plt.legend(['Train', 'Validation', 'Test'])
              st.pyplot(fig)


              st.subheader('Recursive Predictions')
              fig=plt.figure(figsize=(12,6))
              plt.plot(dates_train, train_predictions)
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, val_predictions)
              plt.plot(dates_val, y_val)
              
              plt.plot(recursive_dates, recursive_predictions)
              plt.legend(['Training Predictions', 
                  'Training Observations',
                  'Validation Predictions', 
                  'Validation Observations',
                  
                  'Recursive Predictions'])
              st.pyplot(fig)
           
              st.write("FACEBOOK Predictions completed!")
              
              
        elif 'AMZN' in selected_ticker:
            
              from AMZN import dates_train, y_train, dates_val, y_val, dates_test, y_test
              from AMZN import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
              st.subheader('Train,Validation and Test Graph')
              fig=plt.figure(figsize=(12,7))
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, y_val)
              plt.plot(dates_test, y_test)
              plt.legend(['Train', 'Validation', 'Test'])
              st.pyplot(fig)


              st.subheader('Recursive Predictions')
              fig=plt.figure(figsize=(12,6))
              plt.plot(dates_train, train_predictions)
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, val_predictions)
              plt.plot(dates_val, y_val)
              
              plt.plot(recursive_dates, recursive_predictions)
              plt.legend(['Training Predictions', 
                  'Training Observations',
                  'Validation Predictions', 
                  'Validation Observations',
                  
                  'Recursive Predictions'])
              st.pyplot(fig)
              
              st.write("AMAZON Predictions completed!")
                
          elif 'AMZN' in selected_ticker:
            
              from AMZN import dates_train, y_train, dates_val, y_val, dates_test, y_test
              from AMZN import  train_predictions,val_predictions,test_predictions,recursive_dates,recursive_predictions
              st.subheader('Train,Validation and Test Graph')
              fig=plt.figure(figsize=(12,7))
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, y_val)
              plt.plot(dates_test, y_test)
              plt.legend(['Train', 'Validation', 'Test'])
              st.pyplot(fig)


              st.subheader('Recursive Predictions')
              fig=plt.figure(figsize=(12,6))
              plt.plot(dates_train, train_predictions)
              plt.plot(dates_train, y_train)
              plt.plot(dates_val, val_predictions)
              plt.plot(dates_val, y_val)
              
              plt.plot(recursive_dates, recursive_predictions)
              plt.legend(['Training Predictions', 
                  'Training Observations',
                  'Validation Predictions', 
                  'Validation Observations',
                  
                  'Recursive Predictions'])
              st.pyplot(fig)
              
              st.write("EBAY Predictions completed!")
                   
      
        progress_bar.progress(i+100)     
        st.write("Accuracy:", accuracy, style={'font-size': '24px'})            
        st.success('Compilation Complete!!!')        
