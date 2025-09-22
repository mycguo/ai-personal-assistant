# Add app for adding to the knowledge base
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from simple_vector_store import SimpleVectorStore as MilvusVectorStore
import docx  # Import the python-docx library
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import assemblyai as aai
from moviepy import VideoFileClip
import boto3
import os
import tempfile
import shutil
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from webcrawer import WebCrawler
import yt_dlp as youtube_dl

#configuring the google api key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#tokens from https://www.assemblyai.com/ to transcribe the audio
tokens = st.secrets["ASSEMBLYAI_API_KEY"]

if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Encoding": "identity"  # Disable compression to avoid garbled text
})

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
    metadata_list = []
    for pdf_doc in pdf_docs:
        pdf = PdfReader(pdf_doc)
        # Get document metadata (handle case where metadata is None)
        doc_info = pdf.metadata or {}
        metadata = {
            'filename': pdf_doc.name,
            'num_pages': len(pdf.pages),
            'author': doc_info.get('/Author', 'N/A'),
            'title': doc_info.get('/Title', 'N/A'),
            'subject': doc_info.get('/Subject', 'N/A'),
            'creator': doc_info.get('/Creator', 'N/A'),
            'producer': doc_info.get('/Producer', 'N/A')
        }
        metadata_list.append(metadata)
        
        # Add metadata to the text content
        text += f"\n\nDocument Metadata:\n"
        text += f"Filename: {metadata['filename']}\n"
        text += f"Number of pages: {metadata['num_pages']}\n"
        text += f"Author: {metadata['author']}\n"
        text += f"Title: {metadata['title']}\n"
        text += f"Subject: {metadata['subject']}\n"
        text += f"Creator: {metadata['creator']}\n"
        text += f"Producer: {metadata['producer']}\n"
        text += f"\nDocument Content:\n"
        
        # Extract text from each page
        for page in pdf.pages:
            text += page.extract_text()
    
    return text, metadata_list


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks   

def _load_vector_store(collection_name="personal_assistant"):
    """Load the vector store."""
    try:
        store_path = f"./vector_store_{collection_name}"
        return MilvusVectorStore(store_path=store_path)
    except Exception as e:
        print(f"Error loading vector store: {e}. Creating new store.")
        store_path = f"./vector_store_{collection_name}"
        return MilvusVectorStore.from_texts(
            texts=get_text_chunks("Loading some documents first"),
            store_path=store_path
        )


def _safe_save_vector_store(vector_store, collection_name="personal_assistant"):
    """Milvus automatically persists data, no manual save needed."""
    pass


def get_vector_store(text_chunks):
    vector_store = _load_vector_store()
    vector_store.add_texts(text_chunks)
    return vector_store

def get_current_store():
    return _load_vector_store()

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    return plt

def upload_vector_store_to_s3():
    # Milvus data is managed by the Milvus server
    # For cloud deployment, use Milvus Cloud or Zilliz Cloud
    print("Milvus data is persisted automatically. For cloud deployment, configure Milvus Cloud.")

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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Encoding": "identity"  # Disable compression to avoid garbled text
    }
    # Getting the request from the URL
    r = requests.get(url, headers=headers)
    r.raise_for_status()  # Raise exception for bad status codes
    
    # Handle encoding explicitly
    if r.encoding is None:
        r.encoding = 'utf-8'
        
    # converting the text 
    print(f"Processing url {url}")
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
    st.write("Upload some documents to get started")

   
    st.header("Adding PDF Documents")
    pdf_docs = st.file_uploader("Upload your knowledge base document", type=["pdf"], accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing your PDF documents..."):
            if pdf_docs:
                text, metadata_list = get_pdf_text(pdf_docs)
                
                # Display metadata for each PDF
                for metadata in metadata_list:
                    with st.expander(f"Metadata for {metadata['filename']}"):
                        st.write(f"Number of pages: {metadata['num_pages']}")
                        st.write(f"Author: {metadata['author']}")
                        st.write(f"Title: {metadata['title']}")
                        st.write(f"Subject: {metadata['subject']}")
                        st.write(f"Creator: {metadata['creator']}")
                        st.write(f"Producer: {metadata['producer']}")
                
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                wordcloud_plot = generate_word_cloud(text)
                st.pyplot(wordcloud_plot)
                st.success("Documents processed successfully")

    st.header("Adding Word or Text Documents")
    word_docs = st.file_uploader("Upload your knowledge base document", type=["docx", "txt"], accept_multiple_files=True)
    if st.button("Submit & Process Documents"):
        with st.spinner("Processing your documents..."):
            if word_docs:
                all_files = []
                for doc in word_docs:
                    st.write(f"Processing {doc.name} ... ")
                    if doc.name.lower().endswith(".docx"):
                        # Open the file using docx.Document
                        try:
                            docx_file = docx.Document(doc)
                            # Get document metadata
                            core_properties = docx_file.core_properties
                            metadata = {
                                'filename': doc.name,
                                'author': core_properties.author or 'N/A',
                                'title': core_properties.title or 'N/A',
                                'subject': core_properties.subject or 'N/A',
                                'created': str(core_properties.created) if core_properties.created else 'N/A',
                                'modified': str(core_properties.modified) if core_properties.modified else 'N/A',
                                'last_modified_by': core_properties.last_modified_by or 'N/A'
                            }
                            
                            # Add metadata to the text content
                            text = f"\n\nDocument Metadata:\n"
                            text += f"Filename: {metadata['filename']}\n"
                            text += f"Author: {metadata['author']}\n"
                            text += f"Title: {metadata['title']}\n"
                            text += f"Subject: {metadata['subject']}\n"
                            text += f"Created: {metadata['created']}\n"
                            text += f"Modified: {metadata['modified']}\n"
                            text += f"Last Modified By: {metadata['last_modified_by']}\n"
                            text += f"\nDocument Content:\n"
                            
                            # Add document content
                            paragraphs = [p.text for p in docx_file.paragraphs]
                            text += "\n".join(paragraphs)
                            all_files.append(text)
                            
                            # Display metadata in expander
                            with st.expander(f"Metadata for {metadata['filename']}"):
                                for key, value in metadata.items():
                                    st.write(f"{key.replace('_', ' ').title()}: {value}")
                                    
                        except Exception as e:
                            st.error(f"Error opening the document: {e}")
                            st.stop()
                    elif doc.name.lower().endswith(".txt"):
                        # For text files, add basic metadata
                        metadata = {
                            'filename': doc.name,
                            'type': 'Text File',
                            'size': f"{len(doc.getvalue())} bytes"
                        }
                        
                        # Add metadata to the text content
                        text = f"\n\nDocument Metadata:\n"
                        text += f"Filename: {metadata['filename']}\n"
                        text += f"Type: {metadata['type']}\n"
                        text += f"Size: {metadata['size']}\n"
                        text += f"\nDocument Content:\n"
                        
                        # Add document content
                        text += doc.read().decode("utf-8", errors="replace")
                        all_files.append(text)
                        
                        # Display metadata in expander
                        with st.expander(f"Metadata for {metadata['filename']}"):
                            for key, value in metadata.items():
                                st.write(f"{key.replace('_', ' ').title()}: {value}")
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
                # Read Excel file
                df = pd.read_excel(excel_file)
                
                # Get Excel metadata
                metadata = {
                    'filename': excel_file.name,
                    'type': 'Excel File',
                    'size': f"{len(excel_file.getvalue())} bytes",
                    'sheets': len(df.sheet_names) if hasattr(df, 'sheet_names') else 1,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': ', '.join(df.columns.tolist())
                }
                
                # Add metadata to the text content
                text = f"\n\nDocument Metadata:\n"
                text += f"Filename: {metadata['filename']}\n"
                text += f"Type: {metadata['type']}\n"
                text += f"Size: {metadata['size']}\n"
                text += f"Number of Sheets: {metadata['sheets']}\n"
                text += f"Number of Rows: {metadata['rows']}\n"
                text += f"Number of Columns: {metadata['columns']}\n"
                text += f"Column Names: {metadata['column_names']}\n"
                text += f"\nDocument Content:\n"
                
                # Add Excel content
                text += df.to_string()
                
                # Display metadata in expander
                with st.expander(f"Metadata for {metadata['filename']}"):
                    for key, value in metadata.items():
                        st.write(f"{key.replace('_', ' ').title()}: {value}")
                
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
            with st.expander("URL processed and added"):
                st.text_area("URLs", "\n".join(list(urls)))

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "identity",  # Disable compression to avoid garbled text
                "Connection": "keep-alive"
            }         

            loader = WebBaseLoader(
                    web_path = list(urls),
                    header_template=headers,
                    continue_on_failure = True,
                    show_progress = True)
            all_texts = [doc.page_content for doc in loader.load()]
            text = "\n".join(all_texts)
            # print(text)
            wordcloud_plot = generate_word_cloud(text)
            st.pyplot(wordcloud_plot)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
            st.success("URL processed successfully")         
    
    
    st.header("Audio support")
    audio = st.file_uploader("Update your knowledge base using Audio", type=["mp3"], accept_multiple_files=False)
    if st.button("Submit & Transcribe Audio"):
        with st.spinner("Processing your audio..."):
            if audio:
                st.success("Audio processed successfully")  
                #data = englishTranscription.start_transcription(uploaded_file, tokens)
                transcriber = aai.Transcriber()
                data = transcriber.transcribe(audio)
                with st.expander("View Transcription", expanded=False):
                    st.text_area("Transcription", data.text, height=300)
                wordcloud_plot = generate_word_cloud(data.text)
                st.pyplot(wordcloud_plot)
                st.write("Adding the audio text to the knowledge base")
                text_chunks = get_text_chunks(data.text)
                get_vector_store(text_chunks)
                st.success("Text added to knowledge base successfully")
                
 
    st.header("Video support")
    video = st.file_uploader("Update your knowledge base using Video", type=["mp4"], accept_multiple_files=False)
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
                with st.expander("View Transcription", expanded=False):
                    st.text_area("Transcription", data.text, height=300)
                wordcloud_plot = generate_word_cloud(data.text)
                st.pyplot(wordcloud_plot)
                text_chunks = get_text_chunks(data.text)
                get_vector_store(text_chunks)
                st.success("Text added to knowledge base successfully")
                st.write("")


    st.header("Youtube Video Transcribe")
    st.write("[Note: only work locally because ffmpeg is not avaialbe in the server]")
    link = st.text_input('Enter your YouTube video link', on_change=refresh_state)
    if link:
        #st.video(link)
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
           


    st.write("This is how to setup secrets in streamlit at local environment https://docs.streamlit.io/develop/concepts/connections/secrets-management")
    st.write("This is how to setup secrets in streamlit at cloud https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")

if __name__ == "__main__":
    main()