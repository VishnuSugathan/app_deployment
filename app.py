import numpy as np
import pickle
import pandas as pd
import streamlit as st
import keras
loaded_model = pickle.load(open("C:/Users\imins\Downloads\Tokenizer (1).pickle",'rb'))
def sentiment_prediction(input_data):
    input(input_data)
    prediction = loaded_model.predict(input_data)
    if(prediction[0]==0):
        return 'It is a negative tweet'
    else:
        return 'It is a positive tweet'
    
def main():
    #title
    st.title("Twitter Sentiment Analysis")
    text_in = st.text_input("Enter the text")
    #for prediction
    tweet_pred = ''
    #creating a button for prediction
    if st.button('Tweet analysis result'):
        tweet_pred = sentiment_prediction(text_in)
    
        
    st.success(tweet_pred)
    
if __name__ == '__main__':
    main()
    
    
    
    