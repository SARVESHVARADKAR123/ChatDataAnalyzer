import streamlit as st
import pandas as pd
from src.predictor import ChatPredictor
from src.visualizer import plot_message_counts, plot_wordcloud, plot_suspicious_timeline

st.set_page_config(page_title="ChatLog Investigator", layout="wide")
st.title("💬 ChatLog Investigator - Full Modular Version")

predictor = ChatPredictor()
dataset_path = "data/chat_dataset.csv"

# Session state
if 'last_message' not in st.session_state:
    st.session_state['last_message'] = ""
if 'last_sender' not in st.session_state:
    st.session_state['last_sender'] = ""
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# Sender input
sender = st.text_input("Sender Name", value="Manual")
# Message input
message = st.text_area("Type a message to check if suspicious:")

# Predict
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please type a message first")
    else:
        st.session_state['last_message'] = message
        st.session_state['last_sender'] = sender if sender.strip() != "" else "Manual"
        st.session_state['prediction'] = predictor.predict_messages([message])[0]
        st.success("✅ Prediction ready!")

# Show prediction + flag
if st.session_state['prediction']:
    pred = st.session_state['prediction']
    label = "Suspicious" if pred['is_suspicious'] else "Normal"
    st.write(f"**Prediction:** {label} (Probability: {pred['suspicious_probability']:.2f})")

    if st.button("Flag as Suspicious"):
        predictor.add_and_retrain(
            st.session_state['last_message'],
            sender=st.session_state['last_sender']
        )
        st.success("✅ Message flagged and model retrained")
        # Clear state
        st.session_state['prediction'] = None
        st.session_state['last_message'] = ""
        st.session_state['last_sender'] = ""

# Visualizations
df = pd.read_csv(dataset_path)
st.subheader("📊 Message Counts")
plot_message_counts(df)

st.subheader("☁️ WordCloud of Suspicious Messages")
plot_wordcloud(df)

st.subheader("📅 Suspicious Messages Timeline")
plot_suspicious_timeline(df)
