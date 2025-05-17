import openai
import json
import os
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

# .env içindeki API anahtarını yükle
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📥  Captions dosyasını yükle
with open("captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

# 📦  Embedding sonuçlarını buraya yazacağız
embeddings = {}

# 🔁 Her bir açıklamayı vektörleştir
for url, text in tqdm(captions.items()):
    if text.lower().startswith("error"):
        continue
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        vector = response["data"][0]["embedding"]
        embeddings[url] = vector
    except Exception as e:
        print(f"❌ Error embedding {url}: {e}")

# 💾 Vektörleri kaydet
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ All embeddings saved to embeddings.pkl")
