# UI for asking questions on the knowledge base
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pages.app_admin import get_vector_store, get_text_chunks
from langchain.chains.combine_documents import create_stuff_documents_chain
import boto3
from langchain_nvidia_ai_endpoints import ChatNVIDIA


genai.configure(api_key=os.getenv("GENAI_API_KEY"))
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

nvidia_api_key = st.secrets["NVIDIA_API_KEY"]

def get_prompt_template():
    return PromptTemplate()

def get_chat_chain():
    prompt_template="""
    Answer the questions based on local konwledge base honestly

    Context:\n {context} \n
    Questions: \n {questions} \n

    Answers:
"""
    #model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
    model = ChatNVIDIA(
        model="deepseek-ai/deepseek-r1",
        api_key=nvidia_api_key,
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096
    )
    prompt=PromptTemplate(template=prompt_template, input_variabls=["context","questions"],output_variables=["answers"])
    chain = create_stuff_documents_chain(llm=model, prompt=prompt, document_variable_name="context")
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_chat_chain()

    response = chain.invoke({"context": docs, "questions": user_question})

    print(response)
    st.write("Reply: ",response)

def download_faiss_from_s3():
    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = st.secrets["BUCKET_NAME"]
    print(bucket_name)
    # Download the FAISS index file from S3
    s3.download_file(bucket_name, "index.faiss", "index.faiss")
    s3.download_file(bucket_name, "index.pkl", "index.pkl")
    print(f"Downloaded FAISS index from s3://${bucket_name} to local directory")

def main():
    st.title("Knowledge Assistant")
    st.header("Ask questions on your knowledge base")

    # fix the empty vector store issue
    get_vector_store(get_text_chunks("Loading some documents to build your knowledge base"))

    user_question = st.text_input("Ask me a question")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()