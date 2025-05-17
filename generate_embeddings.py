import openai
import os
import json
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

# 🔐 .env dosyasından API anahtarını al
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📥 Captions dosyasını oku
with open("captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

embeddings = {}

for url, text in tqdm(captions.items()):
    if text.lower().startswith("error"):
        continue
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings[url] = response.data[0].embedding
    except Exception as e:
        print(f"❌ Error embedding {url}: {e}")

# 💾 Kaydet
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Embeddings saved to embeddings.pkl")
