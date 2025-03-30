# Add app for adding to the knowledge base
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import docx  # Import the python-docx library
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import assemblyai as aai
from moviepy import VideoFileClip
import boto3
import os
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

#configuring the google api key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

#tokens from https://www.assemblyai.com/ to transcribe the audio
tokens = st.secrets["ASSEMBLYAI_API_KEY"]

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
    try:
        vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    except Exception:
        vector_store = FAISS.from_texts(get_text_chunks("Loading some documents first"), embedding=embedding)
    vector_store.add_texts(text_chunks)
    vector_store.save_local("faiss_index")
    return vector_store

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    return plt

def upload_vector_store_to_s3():
    # Correct file paths for FAISS index files
    faiss_index_file = "faiss_index/index.faiss"
    faiss_metadata_file = "faiss_index/index.pkl"
   # Ensure the files exist before uploading
    if os.path.exists(faiss_index_file) and os.path.exists(faiss_metadata_file):      
        upload_file_to_s3(faiss_index_file, st.secrets["BUCKET_NAME"], "index.faiss")
        upload_file_to_s3(faiss_metadata_file, st.secrets["BUCKET_NAME"], "index.pkl")
    else:
        print("FAISS index files not found. Ensure they are saved correctly.")

def upload_file_to_s3(local_file_path, bucket_name, s3_key):
    """
    Uploads a file to an S3 bucket.

    :param local_file_path: Path to the local file to upload.
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: Key (path) in the S3 bucket where the file will be stored.
    """
    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    try:
        # Upload the file
        s3.upload_file(local_file_path, bucket_name, s3_key)
        print(f"File {local_file_path} uploaded to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
   
def get_urls(url): 
    urls=[] 
    # getting the request from url 
    r = requests.get(url)      
    # converting the text 
    print(r.text)
    s = BeautifulSoup(r.text,"html.parser")    
    for i in s.find_all("a"):    
        print(i)     
        if 'href' in i.attrs:   
            href = i.attrs['href']            
            if href.startswith("/"):            
                site = url+href 
                print(site)               
                if site not in  urls: 
                    urls.append(site)  
                    print(url) 
    return urls


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
                get_vector_store(text_chunks)
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
                get_vector_store(text_chunks)
                st.success("Documents processed successfully")


    st.header("Adding Excel Documents")
    excel_file = st.file_uploader("Upload your knowledge base document uinsg Excel", type=["xlsx"], accept_multiple_files=False)
    if st.button("Submit & Process Excel"):
        with st.spinner("Processing your excel documents..."):
            if excel_file:
                df = pd.read_excel(excel_file)
                text = df.to_string()
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success("Documents processed successfully")

    st.header("URL fetcher")
    url = st.text_input("Enter the URL")
    if st.button("Submit & Process URL"):
        with st.spinner("Processing your URL..."):
            urls = get_urls(url)
            print(urls)
            for url in urls:
                loader = WebBaseLoader(
                    web_path = url,
                    continue_on_failure = True,
                    show_progress = True
                )
                for doc in loader.load():
                    print(doc.page_content[:100])
                    st.write("Processing link -> ", doc.metadata["source"])
                    text_chunks = get_text_chunks(doc.page_content)
                    get_vector_store(text_chunks)
            st.success("URL processed successfully")

    st.header("Audio support")
    audio = st.file_uploader("Upload your knowledge base document using Audio", type=["mp3"], accept_multiple_files=False)
    if st.button("Submit & Transcribe Audio"):
        with st.spinner("Processing your audio..."):
            if audio:
                st.success("Audio processed successfully")  
                #data = englishTranscription.start_transcription(uploaded_file, tokens)
                transcriber = aai.Transcriber()
                data = transcriber.transcribe(audio)
                #st.write(data.text)
                wordcloud_plot = generate_word_cloud(data.text)
                st.pyplot(wordcloud_plot)
                st.write("Adding the audio text to the knowledge base")
                text_chunks = get_text_chunks(data.text)
                get_vector_store(text_chunks)
                st.success("Text added to knowledge base successfully")
                
 
    st.header("Video support")
    video = st.file_uploader("Upload your knowledge base document using Video", type=["mp4"], accept_multiple_files=False)
    if st.button("Submit & Process Video"):
        with st.spinner("Processing your video..."):
            if video:
                # https://www.bannerbear.com/blog/how-to-use-whisper-api-to-transcribe-videos-python-tutorial/
                bytes_data = video.getvalue()
                with open(video.name, 'wb') as f:
                    f.write(bytes_data)
                st.write("Video file saved successfully!")
                videoClip = VideoFileClip(video.name) 
                audio = videoClip.audio 
                audioFile =video.name.split(".")[0] + ".mp3"
                audio.write_audiofile(audioFile) 
                transcriber = aai.Transcriber()
                data = transcriber.transcribe(audioFile)
                st.write("Adding the audio text to the knowledge base")
                #st.write(data)
                #st.write(data.text)
                wordcloud_plot = generate_word_cloud(data.text)
                st.pyplot(wordcloud_plot)
                text_chunks = get_text_chunks(data.text)
                get_vector_store(text_chunks)
                st.success("Text added to knowledge base successfully")
                st.write("")

    st.write("This is how to setup sercets in streamlit at local environment https://docs.streamlit.io/develop/concepts/connections/secrets-management")
    st.write("This is how to setup sercets in streamlit at cloud https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")

if __name__ == "__main__":
    main()