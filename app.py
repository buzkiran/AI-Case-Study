import streamlit as st
import openai
import faiss
import json
import pickle
import numpy as np
import os
from dotenv import load_dotenv

# Ortam değişkeni
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS verilerini yükle
index = faiss.read_index("faiss_index.index")

with open("faiss_url_map.json", "r", encoding="utf-8") as f:
    url_list = json.load(f)

# Sayfa başlığı
st.set_page_config(page_title="Otel Görsel Arama Sistemi", layout="centered")
st.title("🏨 Otel Odası Görsel Arama")

# Kullanıcıdan sorgu al
query = st.text_input("🔍 Lütfen aramak istediğiniz oda özelliklerini yazınız:", "")

if query:
    with st.spinner("Görseller taranıyor..."):
        try:
            # Embedding üret
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

            # FAISS ile arama
            D, I = index.search(query_vector, 5)

            st.success("📸 En uygun eşleşen görseller:")
            for i in I[0]:
                url = url_list[i]
                st.image(url, width=400, caption=url)
        except Exception as e:
            st.error(f"❌ Hata oluştu: {e}")
