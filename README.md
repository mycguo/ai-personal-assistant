# personal ai chatbot to chat with your personal knowledge base


## run it locally 

```sh
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## System admin Google login

In Streamlit Cloud, open **Manage app → Settings → Secrets** and add the
following configuration alongside the existing API keys. Replace the placeholders
with your Google OAuth web client credentials and a strong random cookie secret.
Keep these values out of Git.

```toml
[auth]
redirect_uri = "https://ai-personal-assistant.streamlit.app/oauth2callback"
cookie_secret = "REPLACE_WITH_A_STRONG_RANDOM_SECRET"

[auth.google]
client_id = "REPLACE_WITH_GOOGLE_OAUTH_CLIENT_ID"
client_secret = "REPLACE_WITH_GOOGLE_OAUTH_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Register the same redirect URI as an authorized redirect URI for your Google
OAuth web client. For local development, use
`http://localhost:8501/oauth2callback` in both places.
The admin page also supports Google credentials directly under `[auth]` without
the `[auth.google]` table. Authentication failures keep admin controls locked.
See [Streamlit authentication configuration](https://docs.streamlit.io/develop/api-reference/user/st.login).

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
