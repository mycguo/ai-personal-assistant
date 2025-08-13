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
import tempfile
import shutil
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

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
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
        # Get document metadata
        doc_info = pdf.metadata
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

def _load_vector_store(path="faiss_index"):
    """Load the FAISS vector store if it exists and is valid."""
    index_file = os.path.join(path, "index.faiss")
    pkl_file = os.path.join(path, "index.pkl")
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading existing index: {e}. Recreating store.")
    return FAISS.from_texts(get_text_chunks("Loading some documents first"), embedding=embedding)


def _safe_save_vector_store(vector_store, path="faiss_index"):
    """Safely save the FAISS vector store by writing to a temporary directory first."""
    tmp_dir = tempfile.mkdtemp()
    try:
        vector_store.save_local(tmp_dir)
        if os.path.exists(path):
            shutil.rmtree(path)
        shutil.move(tmp_dir, path)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def get_vector_store(text_chunks):
    vector_store = _load_vector_store()
    vector_store.add_texts(text_chunks)
    _safe_save_vector_store(vector_store)
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Getting the request from the URL
    r = requests.get(url, headers=headers)       
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
    
    # SSO Authentication support
    st.subheader("Authentication Configuration (Optional)")
    needs_auth = st.checkbox("This site requires authentication")
    
    auth_token = None
    auth_headers = {}
    
    if needs_auth:
        auth_method = st.selectbox("Authentication Method", 
                                 ["Manual Browser (Recommended)", "Browser Extension Helper", "Browser Login (Auto)", "Bearer Token", "API Key", "Cookie", "Custom Header"])
        
        if auth_method == "Manual Browser (Recommended)":
            st.info("🔧 **Manual method - Most reliable for complex sites like Atlassian**")
            st.markdown("""
            **Step-by-step instructions:**
            1. Open your regular browser (Chrome/Firefox/Safari)
            2. Navigate to the login page and complete authentication
            3. Open Developer Tools (F12 or Right-click → Inspect)
            4. Follow the extraction method below
            """)
            
            extraction_method = st.radio("Choose extraction method:", 
                                       ["Cookies (Easy)", "Network Headers (Advanced)", "Local Storage Tokens"])
            
            if extraction_method == "Cookies (Easy)":
                st.markdown("""
                **Extract Cookies:**
                1. In Developer Tools, go to **Application** tab (Chrome) or **Storage** tab (Firefox)
                2. Click on **Cookies** in the left sidebar
                3. Select your domain
                4. Copy all cookie values or use the button below to generate a script
                """)
                
                if st.button("📋 Generate Cookie Extraction Script"):
                    script = f"""
// Run this in browser console (F12 → Console tab) after logging in
let cookies = document.cookie;
console.log("Copy this cookie string:");
console.log(cookies);
// Or copy individual cookies:
document.cookie.split(';').forEach(cookie => console.log(cookie.trim()));
                    """
                    st.code(script, language="javascript")
                
                cookie_input = st.text_area("Paste Cookie String:", 
                                          placeholder="session=abc123; token=xyz789; auth=...",
                                          help="Paste the full cookie string from browser")
                if cookie_input:
                    auth_headers["Cookie"] = cookie_input
                    st.success("✅ Cookies added to headers")
            
            elif extraction_method == "Network Headers (Advanced)":
                st.markdown("""
                **Extract from Network Requests:**
                1. In Developer Tools, go to **Network** tab
                2. Refresh the page or navigate to a protected area
                3. Look for requests to your domain
                4. Right-click a request → **Copy** → **Copy as cURL** or **Copy request headers**
                5. Extract the Authorization header or other auth headers
                """)
                
                header_input = st.text_area("Paste Headers:", 
                                          placeholder="Authorization: Bearer eyJ...\nX-Auth-Token: abc123",
                                          help="Paste headers line by line")
                if header_input:
                    for line in header_input.strip().split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            auth_headers[key.strip()] = value.strip()
                    st.success(f"✅ Added {len(auth_headers)} headers")
                    st.json(list(auth_headers.keys()))
            
            elif extraction_method == "Local Storage Tokens":
                st.markdown("""
                **Extract Tokens from Storage:**
                1. In Developer Tools, go to **Application** tab (Chrome) or **Storage** tab (Firefox)  
                2. Check **Local Storage** and **Session Storage**
                3. Look for tokens (usually keys like 'token', 'access_token', 'auth', 'jwt')
                4. Use the script below to extract all tokens
                """)
                
                if st.button("📋 Generate Token Extraction Script"):
                    script = """
// Run this in browser console (F12 → Console tab) after logging in
console.log("=== LOCAL STORAGE TOKENS ===");
for (let i = 0; i < localStorage.length; i++) {
    let key = localStorage.key(i);
    let value = localStorage.getItem(key);
    if (key.toLowerCase().includes('token') || key.toLowerCase().includes('auth') || key.toLowerCase().includes('jwt')) {
        console.log(`${key}: ${value}`);
    }
}

console.log("=== SESSION STORAGE TOKENS ===");
for (let i = 0; i < sessionStorage.length; i++) {
    let key = sessionStorage.key(i);
    let value = sessionStorage.getItem(key);
    if (key.toLowerCase().includes('token') || key.toLowerCase().includes('auth') || key.toLowerCase().includes('jwt')) {
        console.log(`${key}: ${value}`);
    }
}

console.log("=== ALL COOKIES ===");
console.log(document.cookie);
                    """
                    st.code(script, language="javascript")
                
                token_input = st.text_input("Token Value:", type="password",
                                          help="Paste the token value from localStorage/sessionStorage")
                token_type = st.selectbox("Token Type:", ["Bearer", "API Key", "Custom"])
                
                if token_input:
                    if token_type == "Bearer":
                        auth_headers["Authorization"] = f"Bearer {token_input}"
                    elif token_type == "API Key":
                        header_name = st.text_input("Header Name:", value="X-API-Key")
                        if header_name:
                            auth_headers[header_name] = token_input
                    else:
                        header_name = st.text_input("Header Name:", placeholder="X-Auth-Token")
                        if header_name:
                            auth_headers[header_name] = token_input
                    
                    if auth_headers:
                        st.success("✅ Token added to headers")
        
        elif auth_method == "Browser Extension Helper":
            st.info("🔧 **Browser Extension Method - Copy cookies with one click**")
            st.markdown("""
            **Install a cookie export extension:**
            - **Chrome**: [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm)
            - **Firefox**: [Cookie Quick Manager](https://addons.mozilla.org/en-US/firefox/addon/cookie-quick-manager/)
            - **Edge**: [Cookie-Editor](https://microsoftedge.microsoft.com/addons/detail/cookieeditor/neaplmfkghagebokkhpjpoebhdledlfi)
            
            **Steps:**
            1. Install the extension
            2. Login to your site normally
            3. Click the extension icon
            4. Export/copy all cookies for the domain
            5. Paste below
            """)
            
            export_format = st.selectbox("Export Format:", ["Netscape/curl format", "JSON", "Raw Cookie String"])
            
            cookie_data = st.text_area("Paste Exported Cookies:", 
                                     placeholder="Paste the exported cookie data here...",
                                     height=150)
            
            if cookie_data and st.button("Process Cookie Data"):
                try:
                    if export_format == "JSON":
                        import json
                        cookies = json.loads(cookie_data)
                        if isinstance(cookies, list):
                            cookie_string = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
                        else:
                            cookie_string = "; ".join([f"{k}={v}" for k, v in cookies.items()])
                    elif export_format == "Netscape/curl format":
                        # Parse Netscape format
                        lines = [line.strip() for line in cookie_data.split('\n') if line.strip() and not line.startswith('#')]
                        cookies = []
                        for line in lines:
                            parts = line.split('\t')
                            if len(parts) >= 7:
                                cookies.append(f"{parts[5]}={parts[6]}")
                        cookie_string = "; ".join(cookies)
                    else:
                        cookie_string = cookie_data.strip()
                    
                    auth_headers["Cookie"] = cookie_string
                    st.success("✅ Cookies processed and added")
                    st.text_area("Processed Cookie String:", cookie_string, height=100)
                    
                except Exception as e:
                    st.error(f"Failed to process cookies: {str(e)}")
                    st.info("Try using 'Raw Cookie String' format instead")
        
        elif auth_method == "Browser Login (Auto)":
            st.info("This will open a browser window for you to login, then automatically extract authentication data.")
            login_url = st.text_input("Login URL", value=url, help="URL where you need to login")
            
            browser_choice = st.selectbox("Browser", ["Chrome", "Firefox", "Safari"], 
                                        help="Choose browser for login")
            
            if st.button("🌐 Open Browser & Login"):
                try:
                    # Try undetected-chromedriver first, fallback to regular selenium
                    try:
                        import undetected_chromedriver as uc
                        use_undetected = True
                    except ImportError:
                        from selenium import webdriver
                        use_undetected = False
                        st.info("💡 For better compatibility, install: pip install undetected-chromedriver")
                    
                    import time
                    
                    # Initialize browser based on choice
                    if browser_choice == "Chrome":
                        if use_undetected:
                            # Use undetected-chromedriver for maximum compatibility
                            options = uc.ChromeOptions()
                            options.add_argument("--no-sandbox")
                            options.add_argument("--disable-dev-shm-usage")
                            
                            driver = uc.Chrome(options=options, use_subprocess=True)
                            st.info("✅ Using undetected Chrome driver for better compatibility")
                            
                        else:
                            # Fallback to regular Chrome with anti-detection
                            from selenium.webdriver.chrome.options import Options
                            
                            chrome_options = Options()
                            
                            # Basic stability options
                            chrome_options.add_argument("--no-sandbox")
                            chrome_options.add_argument("--disable-dev-shm-usage")
                            chrome_options.add_argument("--disable-gpu")
                            chrome_options.add_argument("--remote-debugging-port=9222")
                            
                            # Advanced anti-detection options
                            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                            chrome_options.add_experimental_option('useAutomationExtension', False)
                            chrome_options.add_experimental_option("detach", True)
                            
                            # Disable various automation indicators
                            chrome_options.add_argument("--disable-extensions")
                            chrome_options.add_argument("--disable-plugins")
                            chrome_options.add_argument("--disable-javascript-harmony-shipping")
                            chrome_options.add_argument("--disable-xss-auditor")
                            chrome_options.add_argument("--disable-bundled-ppapi-flash")
                            chrome_options.add_argument("--disable-plugins-discovery")
                            chrome_options.add_argument("--disable-prerender-local-predictor")
                            chrome_options.add_argument("--disable-sync")
                            chrome_options.add_argument("--disable-background-timer-throttling")
                            chrome_options.add_argument("--disable-renderer-backgrounding")
                            chrome_options.add_argument("--disable-features=TranslateUI")
                            chrome_options.add_argument("--disable-ipc-flooding-protection")
                            
                            # Set realistic browser profile
                            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                            
                            # Set prefs to avoid detection
                            prefs = {
                                "profile.default_content_setting_values.notifications": 2,
                                "profile.default_content_settings.popups": 0,
                                "profile.default_content_setting_values.plugins": 1,
                            }
                            chrome_options.add_experimental_option("prefs", prefs)
                            
                            driver = webdriver.Chrome(options=chrome_options)
                            
                            # Execute comprehensive anti-detection script
                            driver.execute_script("""
                                // Remove webdriver property
                                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                                
                                // Override the `plugins` property to use a custom getter
                                Object.defineProperty(navigator, 'plugins', {
                                    get: () => [1, 2, 3, 4, 5]
                                });
                                
                                // Override the `languages` property to use a custom getter
                                Object.defineProperty(navigator, 'languages', {
                                    get: () => ['en-US', 'en']
                                });
                                
                                // Mock chrome runtime
                                window.chrome = {
                                    runtime: {}
                                };
                            """)
                        
                    elif browser_choice == "Firefox":
                        from selenium.webdriver.firefox.options import Options
                        firefox_options = Options()
                        firefox_options.add_argument("--disable-blink-features=AutomationControlled")
                        firefox_options.set_preference("dom.webdriver.enabled", False)
                        firefox_options.set_preference("useAutomationExtension", False)
                        firefox_options.set_preference("general.useragent.override", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0")
                        
                        driver = webdriver.Firefox(options=firefox_options)
                        
                    elif browser_choice == "Safari":
                        driver = webdriver.Safari()
                        
                    # Load the page
                    driver.get(login_url)
                    
                    # Wait a moment for the page to load
                    time.sleep(3)
                    
                    st.success("✅ Browser opened! Please complete your login in the browser window.")
                    st.info("After logging in, click 'Extract Auth Data' below to capture cookies and tokens.")
                    
                    # Store driver in session state for later use
                    st.session_state['selenium_driver'] = driver
                    
                except ImportError:
                    st.error("Selenium not installed. Please run: pip install selenium")
                except Exception as e:
                    st.error(f"Failed to open browser: {str(e)}")
                    st.info("💡 **Alternative**: Try opening the site manually in your regular browser, login, then:")
                    st.info("1. Open Developer Tools (F12)")
                    st.info("2. Go to Application/Storage tab")
                    st.info("3. Copy cookies from Cookies section")
                    st.info("4. Use 'Cookie' method above to paste them")
            
            # Extract auth data button
            if 'selenium_driver' in st.session_state and st.button("🔐 Extract Auth Data"):
                try:
                    driver = st.session_state['selenium_driver']
                    
                    # Get all cookies
                    cookies = driver.get_cookies()
                    cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
                    
                    # Try to extract bearer token from localStorage
                    bearer_token = None
                    try:
                        # Common localStorage keys for tokens
                        token_keys = ['token', 'access_token', 'auth_token', 'jwt', 'authToken', 'accessToken']
                        for key in token_keys:
                            token = driver.execute_script(f"return localStorage.getItem('{key}');")
                            if token:
                                bearer_token = token
                                break
                    except:
                        pass
                    
                    # Try to extract from sessionStorage
                    if not bearer_token:
                        try:
                            for key in token_keys:
                                token = driver.execute_script(f"return sessionStorage.getItem('{key}');")
                                if token:
                                    bearer_token = token
                                    break
                        except:
                            pass
                    
                    # Display extracted data
                    st.success("✅ Authentication data extracted!")
                    
                    with st.expander("Extracted Cookies", expanded=True):
                        st.text_area("Cookie String", cookie_string, height=100)
                        if cookie_string:
                            auth_headers["Cookie"] = cookie_string
                    
                    if bearer_token:
                        with st.expander("Extracted Bearer Token", expanded=True):
                            st.text_area("Bearer Token", bearer_token, height=100)
                            auth_headers["Authorization"] = f"Bearer {bearer_token}"
                    
                    # Close browser
                    driver.quit()
                    del st.session_state['selenium_driver']
                    
                except Exception as e:
                    st.error(f"Failed to extract auth data: {str(e)}")
        
        elif auth_method == "Bearer Token":
            auth_token = st.text_input("Bearer Token", type="password", 
                                     help="Enter your OAuth/JWT bearer token")
            if auth_token:
                auth_headers["Authorization"] = f"Bearer {auth_token}"
                
        elif auth_method == "API Key":
            api_key = st.text_input("API Key", type="password")
            key_header = st.text_input("API Key Header Name", value="X-API-Key", 
                                     help="Header name for the API key (e.g., X-API-Key, Authorization)")
            if api_key and key_header:
                auth_headers[key_header] = api_key
                
        elif auth_method == "Cookie":
            cookie_value = st.text_input("Cookie Value", type="password",
                                       help="Enter the full cookie string or session cookie value")
            cookie_name = st.text_input("Cookie Name", value="session",
                                      help="Name of the session cookie (if entering just the value)")
            if cookie_value:
                if "=" in cookie_value:
                    # Full cookie string
                    auth_headers["Cookie"] = cookie_value
                else:
                    # Just cookie value
                    auth_headers["Cookie"] = f"{cookie_name}={cookie_value}"
                    
        elif auth_method == "Custom Header":
            header_name = st.text_input("Header Name", placeholder="e.g., X-Auth-Token")
            header_value = st.text_input("Header Value", type="password")
            if header_name and header_value:
                auth_headers[header_name] = header_value
    
    if st.button("Submit & Process URL"):
        with st.spinner("Processing your URL..."):
            crawler = WebCrawler(url = url, max_depth=max_depth)     
            urls = crawler.start_crawling(url=url)
            with st.expander("URL processed and added"):
                st.text_area("URLs", "\n".join(list(urls)))

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }
            
            # Add authentication headers if provided
            if needs_auth and auth_headers:
                headers.update(auth_headers)
                st.info(f"Added authentication headers: {list(auth_headers.keys())}")

            loader = WebBaseLoader(
                    web_path = list(urls),
                    header_template=headers,
                    continue_on_failure = True,
                    show_progress = True)
            all_texts = [doc.page_content for doc in loader.load()]
            st.write(all_texts);
            text = "\n".join(all_texts)
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