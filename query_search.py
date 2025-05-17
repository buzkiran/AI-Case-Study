import openai
import os
import pickle
import json
import faiss
import numpy as np
from dotenv import load_dotenv

# 🔐 API anahtarını yükle
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📥 FAISS index ve URL listesi
index = faiss.read_index("faiss_index.index")

with open("faiss_url_map.json", "r", encoding="utf-8") as f:
    url_list = json.load(f)

# 🔍 Kullanıcıdan sorgu al
query = input("🔎 Sorgu girin: ")

# ✨ Sorguyu embedding'e çevir
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=query
)
query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

# 🔗 En yakın 5 sonucu getir
D, I = index.search(query_vector, 5)

print("\n🔗 En iyi eşleşen görseller:")
for i in I[0]:
    print("-", url_list[i])
