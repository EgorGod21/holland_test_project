import requests
import numpy as np
from utils.embeddings.config import YANDEX_API_KEY, YANDEX_FOLDER_ID

EMBEDDING_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
HEADERS = {"Authorization": f"Api-key {YANDEX_API_KEY}"}
DOC_URI = f"emb://{YANDEX_FOLDER_ID}/text-search-doc/latest"
QUERY_URI = f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest"

def get_yandex_embedding(text: str, text_type: str = "query") -> np.ndarray:
    data = {
        "modelUri": QUERY_URI if text_type == "query" else DOC_URI,
        "text": text,
    }
    response = requests.post(EMBEDDING_URL, json=data, headers=HEADERS)
    response.raise_for_status()
    return np.array(response.json()["embedding"])
