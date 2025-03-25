# chat-with-charles-resume
# ask any questions about me

## Run it locally

```sh
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

# tech stack
## streamlit: 
web framework
## vector store: 
FAISS (Facebook AI Similarity Search)
## google.generativeai: 
embedding framework, models: "models/embedding-001"
## LangChain: 
Connect LLMs for Retrieval-Augmented Generation (RAG), memory, chaining and agent-based reasoning. 
## PyPDF2 and docx: 
documents import
## assemblyai: 
audio
## moviepy: 
video