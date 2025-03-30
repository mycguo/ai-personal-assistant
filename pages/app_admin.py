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
from webcrawer import WebCrawler
import yt_dlp as youtube_dl

#configuring the google api key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#tokens from https://www.assemblyai.com/ to transcribe the audio
tokens = st.secrets["ASSEMBLYAI_API_KEY"]

if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'

ydl_opts = {
   'format': 'bestaudio/best',
   'postprocessors': [{
       'key': 'FFmpegExtractAudio',
       'preferredcodec': 'mp3',
       'preferredquality': '192',
   }],
   'ffmpeg-location': './',
   'outtmpl': "./%(id)s.%(ext)s",
}

transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
upload_endpoint = 'https://api.assemblyai.com/v2/upload'

headers_auth_only = {'authorization': tokens}
headers = {
   "authorization": tokens,
   "content-type": "application/json"
}
CHUNK_SIZE = 5242880

@st.cache_data
def transcribe_from_link(link, categories: bool):
	_id = link.strip()

	def get_vid(_id):
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			return ydl.extract_info(_id)

	# download the audio of the YouTube video locally
	meta = get_vid(_id)
	save_location = meta['id'] + ".mp3"

	st.write('Saved mp3 to', save_location)

	def read_file(filename):
		with open(filename, 'rb') as _file:
			while True:
				data = _file.read(CHUNK_SIZE)
				if not data:
					break
				yield data


	# upload audio file to AssemblyAI
	upload_response = requests.post(
		upload_endpoint,
		headers=headers_auth_only, data=read_file(save_location)
	)

	audio_url = upload_response.json()['upload_url']
	print('Uploaded to', audio_url)

	# start the transcription of the audio file
	transcript_request = {
		'audio_url': audio_url,
		'iab_categories': 'True' if categories else 'False',
	}

	transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)

	# this is the id of the file that is being transcribed in the AssemblyAI servers
	# we will use this id to access the completed transcription
	transcript_id = transcript_response.json()['id']
	polling_endpoint = transcript_endpoint + "/" + transcript_id

	print("Transcribing at", polling_endpoint)

	return polling_endpoint

def get_status(polling_endpoint):
	polling_response = requests.get(polling_endpoint, headers=headers)
	st.session_state['status'] = polling_response.json()['status']

def refresh_state():
	st.session_state['status'] = 'submitted'



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
    try:
        vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    except Exception:
        vector_store = FAISS.from_texts(get_text_chunks("Loading some documents first"), embedding=embedding)
    vector_store.add_texts(text_chunks)
    vector_store.save_local("faiss_index")
    return vector_store

def get_current_store():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    except Exception:
        vector_store = FAISS.from_texts(get_text_chunks("Loading some documents first"), embedding=embedding)
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

    st.header("Adding Word or Text Documents")
    word_docs = st.file_uploader("Upload your knowledge base document", type=["docx", "txt"], accept_multiple_files=True)
    if st.button("Submit & Process Documents"):
        with st.spinner("Processing your word documents..."):
            if word_docs:
                all_files = []
                for doc in word_docs:
                    st.write(f"Processing {doc.name} ... ")
                    if doc.name.lower().endswith(".docx"):
                        # Open the file using docx.Document
                        try:
                            doc = docx.Document(doc)
                        except Exception as e:
                            st.error(f"Error opening the document: {e}")
                            st.stop()
                        # Example: Extract all paragraphs
                        paragraphs = [p.text for p in doc.paragraphs]
                        #st.write("Paragraphs:")
                        text = "\n".join(paragraphs)
                        all_files.append(text)
                    elif doc.name.lower().endswith(".txt"):
                        text = doc.read().decode("utf-8", errors="replace")
                        all_files.append(text);
                    else:
                        raise NotImplementedError(f"File type {doc.name.split('.')[-1]} not supported")
                all_texts = "\n".join(all_files)
                text_chunks = get_text_chunks(all_texts)
                get_vector_store(text_chunks)
                wordcloud_plot = generate_word_cloud(all_texts)
                st.pyplot(wordcloud_plot)
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
    max_depth = st.number_input("Enter the depth you want to crawel, default is 1, max_value is 3", value=1, max_value=3)
    if st.button("Submit & Process URL"):
        with st.spinner("Processing your URL..."):
            crawler = WebCrawler(url = url, max_depth=max_depth)     
            urls = crawler.start_crawling(url=url)
            print("URL returned")
            print(urls)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }         

            loader = WebBaseLoader(
                    web_path = list(urls),
                    header_template=headers,
                    continue_on_failure = True,
                    show_progress = True)
            all_texts = [doc.page_content for doc in loader.load()]
            text = "\n".join(all_texts)
            wordcloud_plot = generate_word_cloud(text)
            st.pyplot(wordcloud_plot)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
            st.success("URL processed successfully")

    st.header("Youtube Video Transcirbe [Note: only work locally because ffmpeg is not avaialbe in the server]")
    link = st.text_input('Enter your YouTube video link', on_change=refresh_state)
    if link:
        st.video(link)
        st.text("The transcription is " + st.session_state['status'])
        polling_endpoint = transcribe_from_link(link, False)
        st.button('check_status', on_click=get_status, args=(polling_endpoint,))
        transcript=''
        if st.session_state['status']=='completed':
            polling_response = requests.get(polling_endpoint, headers=headers)
            transcript = polling_response.json()['text']

            with st.expander("click to read the content:"):
                st.text_area(transcript)
            wordcloud_plot = generate_word_cloud(transcript)
            st.pyplot(wordcloud_plot)
            st.write("Adding the audio text to the knowledge base")
            text_chunks = get_text_chunks(transcript)
            get_vector_store(text_chunks)
            st.success("Text from Youtube video added to knowledge base successfully")
                    
    
    
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