import openai
import faiss
import json
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS verileri
db = faiss.read_index("faiss_index.index")
with open("faiss_url_map.json", "r", encoding="utf-8") as f:
    url_list = json.load(f)

# Sorgular
def predefined_queries():
    queries = {
        "Sea View Double Room": "double room with sea view",
        "Balcony + AC + City View": "room with balcony, air conditioning, city view",
        "Triple Room with Desk": "triple room that includes a desk",
        "Family Room (4 People)": "room suitable for 4 people with multiple beds"
    }

    for title, query in queries.items():
        print(f"\n🔍 {title}: '{query}'")
        embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        ).data[0].embedding

        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        _, I = db.search(vec, 5)

        for i in I[0]:
            print("→", url_list[i])

if __name__ == "__main__":
    predefined_queries()


# ✅ agents.py — Modüler Ajan Yapısı

class CaptionAgent:
    """GPT-4 Vision ile görsellerden açıklama üretir"""
    def __init__(self, client):
        self.client = client

    def generate_caption(self, image_url):
        # GPT-4 vision çağrısı yapılır
        pass


class EmbeddingAgent:
    """Metinsel açıklamaları vektör haline getirir"""
    def __init__(self, client):
        self.client = client

    def embed(self, text):
        # OpenAI API ile embedding al
        pass


class SearchAgent:
    """Vektörle FAISS üzerinden benzerlik araması yapar"""
    def __init__(self, faiss_index, url_map):
        self.index = faiss_index
        self.urls = url_map

    def search(self, vec, top_k=5):
        # FAISS ile arama yapar
        pass


class OrchestratorAgent:
    """Tüm pipeline'ı yöneten üst seviye yapı"""
    def __init__(self, captioner, embedder, searcher):
        self.captioner = captioner
        self.embedder = embedder
        self.searcher = searcher

    def query_image(self, image_url):
        # Caption → Embedding → FAISS → Sonuç
        pass
