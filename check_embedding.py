import pickle as pkl

with open("embeddings.pkl", "rb") as f:
    data = pkl.load(f)

print("Toplam kayıt:", len(data))
