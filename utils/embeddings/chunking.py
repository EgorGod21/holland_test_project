from typing import List
from transformers import PreTrainedTokenizer

def chunk_text(text: str, max_len: int = 2048, overlap: int = 400) -> List[str]:
    return [text[i:i + max_len] for i in range(0, len(text), max_len - overlap)]

def chunk_text_by_tokens(text: str, tokenizer: PreTrainedTokenizer, max_len: int = 512, overlap: int = 100) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_len - overlap):
        chunk = tokens[i:i + max_len]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

