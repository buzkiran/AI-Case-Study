import openai
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()


# 📌 CaptionAgent: Görselden açıklama üretir
class CaptionAgent:
    def __init__(self, client):
        self.client = client

    def generate_caption(self, image_url):
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this hotel room in detail."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()


# 📌 EmbeddingAgent: Metni vektör haline getirir
class EmbeddingAgent:
    def __init__(self, client):
        self.client = client

    def embed(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype="float32")


# 📌 SearchAgent: FAISS üzerinden benzerlik araması yapar
class SearchAgent:
    def __init__(self, faiss_index_path, url_map_path):
        self.index = faiss.read_index(faiss_index_path)
        with open(url_map_path, "r", encoding="utf-8") as f:
            self.urls = json.load(f)

    def search(self, query_vector, top_k=5):
        vec = query_vector.reshape(1, -1)
        _, I = self.index.search(vec, top_k)
        return [self.urls[i] for i in I[0]]


# 📌 OrchestratorAgent: Tüm ajanları bir araya getirir
class OrchestratorAgent:
    def __init__(self, captioner, embedder, searcher):
        self.captioner = captioner
        self.embedder = embedder
        self.searcher = searcher

    def query_from_text(self, text):
        embedding = self.embedder.embed(text)
        return self.searcher.search(embedding)

    def query_from_image(self, image_url):
        caption = self.captioner.generate_caption(image_url)
        embedding = self.embedder.embed(caption)
        return self.searcher.search(embedding)
