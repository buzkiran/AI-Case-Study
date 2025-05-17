import pickle
import faiss
import numpy as np
import json

# 1. Vektörleri yükle
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# 2. URL'leri ve vektörleri ayır
vectors = []
url_list = []

for url, emb in embeddings.items():
    vectors.append(np.array(emb, dtype="float32"))
    url_list.append(url)

vector_matrix = np.vstack(vectors)  # (25, 1536)

# 3. FAISS index oluştur (L2 mesafe ile)
index = faiss.IndexFlatL2(len(vector_matrix[0]))
index.add(vector_matrix)

# 4. İndeksi ve URL eşlemesini kaydet
faiss.write_index(index, "faiss_index.index")

with open("faiss_url_map.json", "w", encoding="utf-8") as f:
    json.dump(url_list, f, ensure_ascii=False, indent=2)

print("✅ FAISS index and URL map saved.")
