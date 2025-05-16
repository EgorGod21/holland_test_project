import pandas as pd
from tqdm import tqdm
from utils.embeddings.providers.rubert_embedder import get_rubert_embedding

def generate_rubert_embeddings(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    result = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_id = row["user_id"]
        text = str(row["description"])

        try:
            emb = get_rubert_embedding(text)
            result.append({"user_id": user_id, "embeddings": emb.tolist()})
        except Exception as e:
            print(f"Failed for {user_id}: {e}")

    pd.DataFrame(result).to_csv(output_csv, index=False)
