import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

def plot_message_counts(df: pd.DataFrame):
    counts = df['label'].value_counts().rename({0:'Normal',1:'Suspicious'})
    st.bar_chart(counts)

def plot_wordcloud(df: pd.DataFrame):
    suspicious_msgs = df[df['label']==1]['message'].tolist()
    if suspicious_msgs:
        text = " ".join(suspicious_msgs)
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

def plot_suspicious_timeline(df: pd.DataFrame):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    suspicious_df = df[df['label']==1].copy()
    if not suspicious_df.empty:
        timeline = suspicious_df.groupby(suspicious_df['timestamp'].dt.date).size()
        st.line_chart(timeline)
