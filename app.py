import streamlit as st
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import  SentimentIntensityAnalyzer
import dotenv 
import os 
import plotly.express as px
import json
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title('Sentiment Analysis Tool')

st.markdown("Whether it's customer feedback, social media posts, product reviews, \
        or any other form of text, our tool can help you extract valuable insights\
         and understand the underlying sentiment.")


st.subheader('Vadar Sentiment')

text = st.text_input('Enter comment :')
click = st.button('Generate')


dotenv.load_dotenv()
API_TOKEN=os.getenv("HuggingFace")
# print(text)
# txt = 'I love India'
# obj = SentimentIntensityAnalyzer()
# senti_dict = obj.polarity_scores(txt)
# print(senti_dict)

def sentiment(text):
    obj = SentimentIntensityAnalyzer()
    senti_dict = obj.polarity_scores(text)
    if senti_dict['compound']>0.05:
        st.write("😁 Postive")
    elif senti_dict['compound']<=-0.05:
        st.write("Negative :( ")
    else:
        st.write("Neutral !")

if click:
   sentiment(text)

# st.header("Hugging Face Sentiment Analysis ")

st.subheader("Mood Sentiment")


import requests
API_URL =  "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
# API_URL = ""
# headers = {"Authorization": f"Bearer"+st.secrets['HuggingFace']}
# headers = {"Authorization": f"Bearer {st.secrets['HuggingFace']['HuggingFace']}"}

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
# output = query({
#     "inputs": "I like you. I love you",
# })

# print(output)

text2 = st.text_input('Enter comment :',key='txt_input2')
click2 = st.button('Generate',key='btn2')

def sentiment_HF(text):
    try:
        output = query({"inputs":str(text)})
        print(output)
        if text2:
            for data in output:
                print(data)
        # st.write(output[0][0]['label'])
        # print(output)
        label = output[0][0]['label']

        if label == 'positive':
            color = 'pink'
        elif label == 'sadness':
            color = 'pink'
        elif label =='anger':
            color = 'red'
        elif label == 'fear':
            color = 'orange'
        elif label == 'joy':
            color="yellow"
        elif label =="surprise":
            color ="purple"
        elif label =="disgust":
            color ='green'
        else:
            color = 'blue'

        styled_text = f'<h3 style="color:{color}">{label}</h3>'
        st.markdown(styled_text, unsafe_allow_html=True)
        st.markdown(f"The Sentiment is   :red[{output[0][0]['label']}]")
        data = output
        print(data)

        # data = json.loads(output)["data"]
        # st.write(output)
        return output
    except:
      print()
      

# data = json.loads(output)["data"]

# Convert JSON data to a pandas DataFrame
# df = pd.DataFrame(data)
data=sentiment_HF(text2)
print(data)

#   if not df.empty:
if data:
    option = st.selectbox( 'Visualize Insight:',('bar', 'dataframe','scatter'))
    click3 = st.button("Plot Graph",key='click3')
    if click3:
            if option =='bar':
                    df = pd.DataFrame(data[0])
                    fig = px.bar(df, x="label", y="score")
                    st.plotly_chart(fig)
            elif option =='dataframe':
                # df = pd.DataFrame(data[0])
                # st.dataframe(data)   
                df = pd.json_normalize(data[0])
                st.dataframe(df)
            elif option=='scatter':
                df = pd.DataFrame(data[0])
                fig = px.scatter(df, x="label", y="score")
                st.plotly_chart(fig)
        
 
