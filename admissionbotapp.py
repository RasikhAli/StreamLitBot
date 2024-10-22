import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from langdetect import detect
from deep_translator import GoogleTranslator
import pyttsx3
import speech_recognition as sr
from twilio.rest import Client  # For WhatsApp integration

# WhatsApp configurations
TWILIO_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_WHATSAPP_NUMBER = ''

# Load the admissions queries dataset and embeddings
admission_df = pd.read_csv('data/admission_queries_responses4.csv')
admission_embeddings = np.load('data/admission_query_embeddings.npy')

# Load the SentenceTransformer model for admission queries
model_path_admission = 'LLM/paraphrase-MiniLM-L6-v2'
model_admission = SentenceTransformer(model_path_admission)

# Load FAISS index for admission queries
index_admission = faiss.read_index('data/admission_faiss.index')

# Initialize the TTS engine (Text-to-Speech)
engine = pyttsx3.init()

# Set properties for voice
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Function to detect the language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# Function to retrieve similar admissions queries
def retrieve_similar_queries(query, model, index, df, k=5):
    query_embedding = model.encode([clean_text(query)])
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i in range(k):
        index_pos = indices[0][i]
        context_text = df['Context'].iloc[index_pos] if not pd.isna(df['Context'].iloc[index_pos]) else 'N/A'
        query_text = df['Query'].iloc[index_pos] if not pd.isna(df['Query'].iloc[index_pos]) else 'N/A'
        response_text = df['Response'].iloc[index_pos] if not pd.isna(df['Response'].iloc[index_pos]) else 'N/A'
        
        query_info = {
            'query': query_text,
            'response': response_text,
            'distance': float(distances[0][i]),
            'context': context_text
        }
        results.append(query_info)

    if not results:
        return [{'response': "Couldn't find matching queries."}]

    return results

# Streamlit UI
st.title("Admission Query Chatbot")

# User Input: Text Box
query = st.text_input("Enter your query:")

# Handle the query and provide results
if st.button('Submit Query'):
    if query:
        lang = detect_language(query)
        if lang == 'ar':
            translated_query = GoogleTranslator(source='ar', target='en').translate(query)
            results = retrieve_similar_queries(translated_query, model_admission, index_admission, admission_df, k=1)
        elif lang == 'ur':
            translated_query = GoogleTranslator(source='ur', target='en').translate(query)
            results = retrieve_similar_queries(translated_query, model_admission, index_admission, admission_df, k=1)
            for result in results:
                result['response'] = GoogleTranslator(source='en', target='ur').translate(result['response'])
        else:
            results = retrieve_similar_queries(query, model_admission, index_admission, admission_df, k=1)
        
        # Display results
        if results:
            st.write("Here are the results:")
            for result in results:
                st.write(f"Query: {result['query']}")
                st.write(f"Response: {result['response']}")
        else:
            st.write("No similar queries found.")
    else:
        st.warning("Please enter a query.")

# WhatsApp Integration
st.write("Send the response via WhatsApp:")
whatsapp_number = st.text_input("Enter your WhatsApp number (with country code):")

if st.button('Send on WhatsApp'):
    if whatsapp_number:
        # Twilio API Call to Send WhatsApp message
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        try:
            # Example message to be sent (customize based on query results)
            if results and len(results) > 0:
                message_text = f"Assalam o Alaikum,\nYour query: {results[0]['query']}\nResponse: {results[0]['response']}\nJazakAllah, Allah Hafiz."
            else:
                message_text = "No relevant response found. Please try a different query."
                
            message = client.messages.create(
                body=message_text,
                from_=TWILIO_WHATSAPP_NUMBER,
                to=f'whatsapp:+{whatsapp_number}'
            )
            st.success(f"Message sent to {whatsapp_number}")
        except Exception as e:
            st.error(f"Failed to send WhatsApp message. Error: {e}")
    else:
        st.warning("Please enter your WhatsApp number.")
