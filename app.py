import streamlit as st
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import  SentimentIntensityAnalyzer
import dotenv 
import os 
import plotly.express as px
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
        st.markdown("😁 yellow[Postive]")
    elif senti_dict['compound']<=-0.05:
        st.write("red[Negative] :( ")
    else:
        st.write("green[Neutral]")

if click:
   sentiment(text)

# st.header("Hugging Face Sentiment Analysis ")

st.subheader("Mood Sentiment")


import requests
API_URL =  "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

# headers = {"Authorization": f"Bearer"+st.secrets['HuggingFace']}
headers = {"Authorization": f"Bearer {st.secrets['HuggingFace']}"}

# headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
# output = query({
#     "inputs": "I like you. I love you",
# })

# print(output)

text2 = st.text_input('Enter comment :',key='txt_input2')
click2 = st.button('Generate & Visualize',key='btn2')

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
      

data=sentiment_HF(text2)
print(data)


if click2:
   try:
        df = pd.DataFrame(data[0]) 
        df = df.rename(columns={"label": "emotion"})     
        st.subheader("Bar Chart")              
        fig = px.bar(df, x="emotion", y="score")
        st.plotly_chart(fig)
                

        st.subheader("Line Chart")         
        fig = px.line(df, x="emotion", y="score")
        st.plotly_chart(fig)     
        
        st.subheader("Dataframe")              
        st.dataframe(df)

        st.subheader("Scatter Chart")         
        fig = px.scatter(df, x="emotion", y="score")
        st.plotly_chart(fig)

        st.subheader("Pie Chart")         
        fig = px.pie(df, names="emotion", values="score")
        st.plotly_chart(fig)
                
        
      
   except:
       print()