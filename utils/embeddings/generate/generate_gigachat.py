import pandas as pd
import time
from tqdm import tqdm
from utils.embeddings.providers.gigachat_embedder import get_gigachat_embedding

MAX_REQUESTS_PER_SECOND = 10
REQUEST_DELAY = 1 / MAX_REQUESTS_PER_SECOND

def generate_gigachat_embeddings(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    result = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_id = row["user_id"]
        text = str(row["description"])

        try:
            emb = get_gigachat_embedding(text)
            result.append({"user_id": user_id, "embeddings": emb.tolist()})
        except Exception as e:
            print(f"Failed for {user_id}: {e}")

        time.sleep(REQUEST_DELAY)

    pd.DataFrame(result).to_csv(output_csv, index=False)
