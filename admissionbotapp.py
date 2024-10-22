import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS  # Google Text-to-Speech
import os  # For file handling
import speech_recognition as sr

# Load the admissions queries dataset and embeddings
admission_df = pd.read_csv('data/admission_queries_responses4.csv')
admission_embeddings = np.load('data/admission_query_embeddings.npy')

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model_admission = load_model()

# Load FAISS index for admission queries
index_admission = faiss.read_index('data/admission_faiss.index')

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

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service")
    return None

# Start/Stop conversation buttons
if st.button("Start Conversation"):
    while True:
        # Recognize speech from the microphone
        query = recognize_speech()
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
                for result in results:
                    st.write(f"Query: {result['query']}")
                    st.write(f"Response: {result['response']}")
                    # Convert the response to speech
                    tts = gTTS(text=result['response'], lang='en')
                    audio_file = f"response_{result['query']}.mp3"
                    tts.save(audio_file)
                    st.audio(audio_file)  # Play the audio

            else:
                st.write("No similar queries found.")

        if st.button("End Conversation"):
            st.write("Conversation ended.")
            break

else:
    # User Input: Text Box (for manual queries)
    query = st.text_input("Or, enter your query manually:")

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
                for result in results:
                    st.write(f"Query: {result['query']}")
                    st.write(f"Response: {result['response']}")
                    # Convert the response to speech
                    tts = gTTS(text=result['response'], lang='en')
                    audio_file = f"response_{result['query']}.mp3"
                    tts.save(audio_file)
                    st.audio(audio_file)  # Play the audio
            else:
                st.write("No similar queries found.")
        else:
            st.warning("Please enter a query.")
