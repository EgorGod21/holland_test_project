import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils.embeddings.chunking import chunk_text_by_tokens

tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
model = AutoModel.from_pretrained("ai-forever/ruBert-base")
model.eval()
if torch.cuda.is_available():
    model.cuda()

@torch.no_grad()
def get_rubert_embedding(text: str) -> np.ndarray:
    chunks = chunk_text_by_tokens(text, tokenizer, max_len=512, overlap=100)
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        output = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(output.cpu().numpy()[0])
    return np.median(embeddings, axis=0)
