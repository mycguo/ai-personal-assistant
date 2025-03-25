# UI for asking questions on the knowledge base
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import StuffDocumentsChain, LLMChain  # Updated import
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pages.app_admin import get_vector_store, get_text_chunks

load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))


def get_prompt_template():
    return PromptTemplate()

def get_chat_chain():
    prompt_template="""
    Answer the questions based on local konwledge base honestly

    Context:\n {context} \n
    Questions: \n {questions} \n

    Answers:
"""
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variabls=["context","questions"],output_variables=["answers"])
    llm_chain = LLMChain(llm=model, prompt=prompt)  # Create LLMChain instance
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")  # Pass LLMChain instance and specify document_variable_name
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_chat_chain()

    response = chain({"input_documents": docs, "questions": user_question}, return_only_outputs=True)              

    print(response)
    st.write("Reply: ",response["output_text"])


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