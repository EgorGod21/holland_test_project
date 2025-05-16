import numpy as np
from gigachat import GigaChat
from config import GIGACHAT_TOKEN
from utils.embeddings.chunking import chunk_text

def get_gigachat_embedding(text: str) -> np.ndarray:
    chunks = chunk_text(text, max_len=1800, overlap=400)
    embeddings = []
    with GigaChat(credentials=GIGACHAT_TOKEN, verify_ssl_certs=False) as giga:
        for chunk in chunks:
            emb = giga.embeddings([chunk]).data[0].embedding
            embeddings.append(np.array(emb))
    return np.mean(embeddings, axis=0)
