# UI for asking questions on the knowledge base
import streamlit as st
import os
import google.generativeai as genai
from file_manager import get_all_file_uris

# Configure API key
# Configure API key
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
elif "GENAI_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GENAI_API_KEY"))
else:
    st.error("GOOGLE_API_KEY not found in secrets or environment.")

def get_model():
    model_name = st.sidebar.selectbox(
        "Select Model", 
        ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-3-pro-preview"],
        index=0
    )
    return genai.GenerativeModel(model_name)

def user_input(user_question):
    file_uris = get_all_file_uris()
    
    if not file_uris:
        st.warning("No documents found in the knowledge base. Please upload documents in the Admin page.")
        return

    model = get_model()
    
    # Construct the prompt with files
    # For Gemini, we can pass file URIs directly in the content
    content = []
    
    # Add files to context
    for uri in file_uris:
        file_obj = genai.get_file(uri) # Verify it exists/get metadata if needed, or just pass URI object if supported
        # The python SDK allows passing the file object returned by upload_file or get_file, 
        # or a part object. Let's get the file object.
        content.append(file_obj)
        
    content.append(user_question)

    import time
    from google.api_core import exceptions

    with st.spinner("Thinking..."):
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                response = model.generate_content(content)
                st.write("Reply:", response.text)
                break
            except exceptions.ResourceExhausted as e:
                retry_count += 1
                wait_time = 2 ** retry_count + 10 # Exponential backoff + buffer
                if retry_count == max_retries:
                    st.error(f"Quota exceeded after {max_retries} retries. Please try a different model or wait longer.")
                else:
                    st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                break

def main():
    st.title("AI Knowledge Assistant (Gemini 3 + File Search)")
    st.header("Ask questions on your knowledge base")

    user_question = st.text_input("Ask me a question about your documents:")
    if user_question:
        user_input(user_question)
    
    st.markdown("<div style='height:300px;'></div>", unsafe_allow_html=True)
    st.markdown(""" 
        # Tech Stack
        - **Web Framework**: Streamlit
        - **Model**: Gemini 3.0 (gemini-2.0-flash-exp as placeholder if 3.0 not avail)
        - **RAG**: Google Gen AI File API (Long Context)
    """)    

if __name__ == "__main__":
    main()