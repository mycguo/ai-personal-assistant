import importlib
import sys
from types import ModuleType
from unittest.mock import patch, MagicMock

_dummy_nvidia = ModuleType('langchain_nvidia_ai_endpoints')
_dummy_nvidia.ChatNVIDIA = MagicMock()
_dummy_genai = ModuleType('google.generativeai')
_dummy_genai.configure = MagicMock()
_dummy_lgg = ModuleType('langchain_google_genai')
_dummy_lgg.GoogleGenerativeAIEmbeddings = MagicMock(return_value=object())
_dummy_lgg.ChatGoogleGenerativeAI = MagicMock()
_dummy_assembly = ModuleType('assemblyai')


def test_get_prompt_template():
    mock_prompt = MagicMock(return_value='tmpl')
    with patch.dict(sys.modules, {
        'langchain_nvidia_ai_endpoints': _dummy_nvidia,
        'google.generativeai': _dummy_genai,
        'langchain_google_genai': _dummy_lgg,
        'assemblyai': _dummy_assembly,
    }), \
         patch('streamlit.secrets', {"NVIDIA_API_KEY": "dummy", "GOOGLE_API_KEY": "dummy", "ASSEMBLYAI_API_KEY": "dummy"}), \
         patch('langchain_community.vectorstores.FAISS', MagicMock()), \
         patch('langchain.prompts.PromptTemplate', mock_prompt):
        app = importlib.import_module('app')
        template = app.get_prompt_template()
    assert template == 'tmpl'
    mock_prompt.assert_called_once()


def test_get_text_chunks_split():
    with patch.dict(sys.modules, {
        'langchain_nvidia_ai_endpoints': _dummy_nvidia,
        'google.generativeai': _dummy_genai,
        'langchain_google_genai': _dummy_lgg,
        'assemblyai': _dummy_assembly,
    }), \
         patch('streamlit.secrets', {"GOOGLE_API_KEY": "dummy", "ASSEMBLYAI_API_KEY": "dummy"}), \
         patch('langchain_community.vectorstores.FAISS', MagicMock()):
        pages = importlib.import_module('pages.app_admin')
        get_text_chunks = pages.get_text_chunks
    text = 'A' * 6000
    chunks = get_text_chunks(text)
    assert len(chunks) >= 2

