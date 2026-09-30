from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from app.config import EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

chat_model = None
embedding_model = None


def get_chat_model() -> ChatOpenAI:
    global chat_model
    if chat_model is None:
        chat_model = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
        )
    return chat_model


def get_embeddings() -> HuggingFaceEmbeddings:
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embedding_model
