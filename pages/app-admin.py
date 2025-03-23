# Add app for adding to the knowledge base
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import docx  # Import the python-docx library
import pandas as pd
import requests
import torch
import whisper  
import numpy as np
from io import BytesIO
from pydub import AudioSegment


#configuring the google api key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf = PdfReader(pdf_doc)
        for page in pdf.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks   

def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embedding)

    vector_store.save_local("faiss_index")
    return vector_store

def main():
    st.title("Knowledge Assistant")
    st.header("Adding Documents to your knowledge base")
    st.write("Upload your knowledge base documents to get started")

   
    st.header("Adding PDF Documents")
    pdf_docs = st.file_uploader("Upload your knowledge base document", type=["pdf"], accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing your PDF documents..."):
            if pdf_docs:
                text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text)
                vector_store = get_vector_store(text_chunks)
                st.success("Documents processed successfully")

    st.header("Adding Word Documents")
    word_docs = st.file_uploader("Upload your knowledge base document", type=["docx"], accept_multiple_files=False)
    if st.button("Submit & Process Word"):
        with st.spinner("Processing your word documents..."):
            if word_docs:
                # Open the file using docx.Document
                try:
                    doc = docx.Document(word_docs)
                except Exception as e:
                    st.error(f"Error opening the document: {e}")
                    st.stop()
                # Example: Extract all paragraphs
                paragraphs = [p.text for p in doc.paragraphs]
                #st.write("Paragraphs:")
                text = "\n".join(paragraphs)
                text_chunks = get_text_chunks(text)
                vector_store = get_vector_store(text_chunks)
                st.success("Documents processed successfully")


    st.header("Adding Excel Documents")
    excel_file = st.file_uploader("Upload your knowledge base document uinsg Excel", type=["xlsx"], accept_multiple_files=False)
    if st.button("Submit & Process Excel"):
        with st.spinner("Processing your excel documents..."):
            if excel_file:
                df = pd.read_excel(excel_file)
                text = df.to_string()
                text_chunks = get_text_chunks(text)
                vector_store = get_vector_store(text_chunks)
                st.success("Documents processed successfully")

    st.header("URL fetcher")
    url = st.text_input("Enter the URL")
    if st.button("Submit & Process URL"):
        with st.spinner("Processing your URL..."):
            if url:
                response = requests.get(url)
                text = response.text
                text_chunks = get_text_chunks(text)
                vector_store = get_vector_store(text_chunks)
                st.success("URL processed successfully")

    st.header("Audio support")
    audio = st.file_uploader("Upload your knowledge base document using Audio", type=["mp3"], accept_multiple_files=False)
    if st.button("Submit & Process Audio"):
        with st.spinner("Processing your audio..."):
            if audio:
                st.success("Audio processed successfully")  
                model = whisper.load_model("base")
                audio_bytes = audio.read()
                audio_file = BytesIO(audio_bytes)
                audio_segment = AudioSegment.from_file(audio_file, format="mp3")
                audio_segment.export(audio_file, format="wav")
                audio_file.seek(0)
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
                result = model.transcribe(audio_array)                
                st.write("Adding the audio text to the knowledge base")
                st.write(result["text"])
                text_chunks = get_text_chunks(result["text"])
                vector_store = get_vector_store(text_chunks)
                st.success("Text added to knowledge base successfully")
                
 
    st.header("Video support")
    video = st.file_uploader("Upload your knowledge base document using Video", type=["mp4"], accept_multiple_files=False)
    if st.button("Submit & Process Video"):
        with st.spinner("Processing your video..."):
            if video:
                st.success("Video processed successfully")  
                model = whisper.load_model("base")
                video_bytes = video.read()
                audio = np.frombuffer(video_bytes, np.int16).astype(np.float32) / 32768.0
                result = model.transcribe(audio)
                st.write("Adding the audio text to the knowledge base")
                st.write(result["text"])
                text_chunks = get_text_chunks(result["text"])
                vector_store = get_vector_store(text_chunks)
                st.success("Text added to knowledge base successfully")
                st.write("")

    st.write("This is how to setup sercets in streamlit at local environment https://docs.streamlit.io/develop/concepts/connections/secrets-management")
    st.write("This is how to setup sercets in streamlit at cloud https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
if __name__ == "__main__":
    main()