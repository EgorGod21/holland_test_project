import pandas as pd
import time
from tqdm import tqdm
from utils.embeddings.providers.yandex_embedder import get_yandex_embedding

def generate_yandex_embeddings(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    result = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            emb = get_yandex_embedding(row["description"])
            result.append({"user_id": row["user_id"], "embeddings": emb.tolist()})
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed for {row['user_id']}: {e}")
    pd.DataFrame(result).to_csv(output_csv, index=False)
