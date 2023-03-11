# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:46:48 2023

@author: Dell
"""

import re
import pickle
import pandas as pd
import streamlit as st

from tensorflow import keras
from keras import backend as K

from keras import models
from keras.models import load_model
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer


# path of the model
#MODEL_PATH = "Sentiment-BiLSTM.h5"
# maximum number of the allowed word in an input 
max_words = 500
# shape of input data passed for prediction
max_len = 100
# path of tokenizer file
tokenizer_file = "Tokenizer.pickle"

# load tokenizer
with open(tokenizer_file,'rb') as handle:
    tokenizer = pickle.load(handle)

contractions = pd.read_csv("https://drive.google.com/file/d/1FIRgp3lp5FY6wescbm4gCsCzonwf2q8Z/view?usp=share_link", index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Defining regex patterns.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"




def preprocess_apply(tweet):

    tweet = tweet.lower()
        
    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'', tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

     

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)

    # Remove numbers.
    tweet = re.sub('[0-9]+', '', tweet)
    return tweet

# load the sentiment analysis model
@st.cache(allow_output_mutation=True)
def Load_model():
    model = load_model("https://drive.google.com/file/d/1-gZ819-UGMxWcrYpAi81zYJ27jpq3B1V/view?usp=share_link")
    model.summary() # included making it visible when the model is reloaded
    session = K.get_session()
    return model, session

if __name__ == '__main__':
    st.title('Twitter Sentimental analysis')
    st.write('A polarity classification application')
    st.subheader('Input the Tweet in the below space')
    sentence = st.text_area('Enter your tweet here',height=200)
    predict_btt = st.button('predict')
    model, session = Load_model()
    if predict_btt:
        clean_text = []
        K.set_session(session)
        i = preprocess_apply(sentence)
        clean_text.append(i)
        sequences = tokenizer.texts_to_sequences(clean_text)
        data = pad_sequences(sequences, maxlen =  max_len)
        # st.info(data)
        prediction = model.predict(data)
        
        
        
        
        st.header('Prediction using LSTM model')
        if prediction < 0.5:
          st.warning('Tweet has negative sentiment')
        else :
          st.success('Tweet has positive sentiment')
        
