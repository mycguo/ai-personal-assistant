# Add app for adding to the knowledge base
import streamlit as st
import google.generativeai as genai
import os
import tempfile
from file_manager import save_file_entry, load_files, clear_files
import time

# configuring the google api key
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("GOOGLE_API_KEY not found in secrets.")

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    st.write(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    st.write("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    st.write("All files are ready.")

def main():
    st.title("Knowledge Assistant Admin")
    st.header("Upload Documents to Gemini 3 Context")
    st.write("Upload documents to be used by the Gemini 3 model for RAG.")

    # Display currently uploaded files
    st.subheader("Currently Uploaded Files")
    current_files = load_files()
    if current_files:
        for f in current_files:
            st.text(f"- {f['name']} ({f['mime_type']})")
        
        if st.button("Clear All Files"):
            clear_files()
            st.rerun()
    else:
        st.info("No files uploaded yet.")

    st.header("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Text, CSV, etc.)", 
        type=["pdf", "txt", "csv", "md", "py", "json"], 
        accept_multiple_files=True
    )

    if st.button("Upload & Process"):
        if uploaded_files:
            with st.spinner("Uploading to Gemini..."):
                uploaded_gemini_files = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Determine mime type (basic)
                        mime_type = "text/plain"
                        if uploaded_file.name.lower().endswith(".pdf"):
                            mime_type = "application/pdf"
                        elif uploaded_file.name.lower().endswith(".csv"):
                            mime_type = "text/csv"
                        
                        # Upload to Gemini
                        gemini_file = upload_to_gemini(tmp_path, mime_type=mime_type)
                        uploaded_gemini_files.append(gemini_file)
                        
                        # Save to local registry
                        save_file_entry(uploaded_file.name, gemini_file.name, mime_type) # gemini_file.name is the ID/URI part usually
                        
                    finally:
                        os.remove(tmp_path)
                
                # Wait for processing
                try:
                    wait_for_files_active(uploaded_gemini_files)
                    st.success("All files uploaded and processed successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error waiting for file processing: {e}")

if __name__ == "__main__":
    main()
