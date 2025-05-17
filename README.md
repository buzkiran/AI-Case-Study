# 🧠 Otel Görseli ile Arama Sistemi – oBilet Jr. AI Specialist Case Study

Bu proje, **Aybüke Sude Buzkıran** tarafından oBilet Jr. AI Specialist pozisyonu için geliştirilmiştir.  
Amaç, otel görsellerinden yapay zeka ile açıklama üretmek, bu açıklamaları vektörleştirmek ve kullanıcıdan gelen metin sorgularına göre en uygun görselleri döndürmektir.

---

## 🔧 Kullanılan Teknolojiler

- OpenAI GPT-4 Turbo (Vision desteği ile) – Görselden açıklama üretimi
- OpenAI `text-embedding-ada-002` – Açıklamaların vektöre çevrilmesi
- FAISS – Hızlı benzerlik arama algoritması
- Python, NumPy, Requests, Pickle, dotenv, tqdm

---

## 🗂️ Proje Dosya Yapısı

```
AI-Case-Study/
├── generate_captions.py         # Görsellerden açıklama üretir (captioning)
├── captions.json                # URL → açıklama eşleşmeleri
├── generate_embeddings.py       # Açıklamaları embedding'e çevirir
├── embeddings.pkl               # Pickle formatında vektör verisi
├── build_faiss_index.py         # FAISS ile vektör indeks oluşturur
├── faiss_index.index            # FAISS ikili indeks dosyası
├── faiss_url_map.json           # İndeks numarası → URL eşleşmeleri
├── query_search.py              # Kullanıcıdan gelen sorguya karşılık benzer görselleri bulur
├── .env                         # OpenAI API anahtarı (gizli)
├── requirements.txt             # Python bağımlılıkları
```

---

## ▶️ Kurulum ve Kullanım

### 1. Ortamı Oluştur

```bash
python -m venv case
case\Scripts\activate
pip install -r requirements.txt
```

### 2. API Anahtarı Ekle

Proje dizinine `.env` dosyası oluşturun:

```
OPENAI_API_KEY=sk-...
```

### 3. Adım Adım Çalıştır

```bash
python generate_captions.py         # Açıklama üretimi
python generate_embeddings.py       # Vektör üretimi
python build_faiss_index.py         # FAISS indeks oluşturma
python query_search.py              # Sorgu çalıştırma
```

---

## 🔍 Örnek Sorgu

```
🔎 Sorgu girin: deniz manzaralı balkonlu çift kişilik oda
```

```
🔗 En iyi eşleşen görseller:
- https://static.obilet.com.../1.jpg
- https://static.obilet.com.../7.jpg
...
```

---

## 🔐 Güvenlik Notu

- `requests.get(..., verify=False)` tercihi, yalnızca sabit AWS görsel linklerinde karşılaşılan SSL hatalarını geçici olarak aşmak için kullanılmıştır. Bu bilinçli ve sınırlı bir tercihtir.
- OpenAI API anahtarı `.env` içinde saklanır ve `.gitignore` ile versiyon kontrolüne alınmaz.
- Yanlışlıkla commit edilen bir anahtar GitHub Push Protection tarafından engellenmiş ve `git filter-repo` ile geçmişten silinmiştir.

---

## 👩‍💻 Geliştirici

**Aybüke Sude Buzkıran**  
Yapay Zeka Mühendisi  
İstanbul / Beşiktaş

---

## 🙏 Teşekkür

OpenAI’nin sunduğu GPT-4 Vision ve embedding modelleri sayesinde bu sistem başarıyla hayata geçirilmiştir.  
Bu belge, proje sürecinde karşılaşılan teknik zorlukları ve çözümleri de belgeler.