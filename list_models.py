import google.generativeai as genai
import os
import streamlit as st

# Try to get key from secrets or env
api_key = os.environ.get("GENAI_API_KEY")
if not api_key and os.path.exists(".streamlit/secrets.toml"):
    # This is a hacky way to read secrets if not running in streamlit, 
    # but for this script I'll just rely on the user having the env var set 
    # or I will try to read the file manually if needed.
    # Actually, let's just assume the environment has it or we can't run this script easily.
    pass

if not api_key:
    print("No API key found in env. Please set GENAI_API_KEY.")
else:
    genai.configure(api_key=api_key)
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
