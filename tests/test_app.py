import importlib
import sys
from types import ModuleType
from unittest.mock import patch, MagicMock

_dummy_nvidia = ModuleType('langchain_nvidia_ai_endpoints')
_dummy_nvidia.ChatNVIDIA = MagicMock()
_dummy_genai = ModuleType('google.generativeai')
_dummy_genai.configure = MagicMock()
_dummy_genai.GenerativeModel = MagicMock()
_dummy_lgg = ModuleType('langchain_google_genai')
_dummy_lgg.GoogleGenerativeAIEmbeddings = MagicMock(return_value=object())
_dummy_lgg.ChatGoogleGenerativeAI = MagicMock()
_dummy_assembly = ModuleType('assemblyai')


def test_get_model_uses_selected_gemini_model():
    _dummy_genai.GenerativeModel.reset_mock()
    with patch.dict(sys.modules, {
        'langchain_nvidia_ai_endpoints': _dummy_nvidia,
        'google.generativeai': _dummy_genai,
        'langchain_google_genai': _dummy_lgg,
        'assemblyai': _dummy_assembly,
    }), \
         patch('streamlit.secrets', {"NVIDIA_API_KEY": "dummy", "GOOGLE_API_KEY": "dummy", "ASSEMBLYAI_API_KEY": "dummy"}), \
         patch('streamlit.sidebar.selectbox', return_value='gemini-2.0-flash-exp'):
        sys.modules.pop('app', None)
        app = importlib.import_module('app')
        app.get_model()
    _dummy_genai.GenerativeModel.assert_called_once_with('gemini-2.0-flash-exp')


def test_get_text_chunks_split():
    with patch.dict(sys.modules, {
        'langchain_nvidia_ai_endpoints': _dummy_nvidia,
        'google.generativeai': _dummy_genai,
        'langchain_google_genai': _dummy_lgg,
        'assemblyai': _dummy_assembly,
    }), \
         patch('streamlit.secrets', {"GOOGLE_API_KEY": "dummy", "ASSEMBLYAI_API_KEY": "dummy"}), \
         patch('langchain_community.vectorstores.FAISS', MagicMock()):
        sys.modules.pop('pages.app_admin', None)
        pages = importlib.import_module('pages.app_admin')
        get_text_chunks = pages.get_text_chunks
    text = 'A' * 6000
    chunks = get_text_chunks(text)
    assert len(chunks) >= 2
