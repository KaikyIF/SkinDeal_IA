import sys
import torch
import open_clip
import numpy as np
import os


from PIL import Image

if len(sys.argv) < 2:
    print("Uso: python search_image.py imagem.jpg")
    sys.exit()

if not os.path.exists(sys.argv[1]):
    print(f"Arquivo não encontrado: {sys.argv[1]}")
    sys.exit()

# carregar modelo
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model.eval()

# carregar índice
embeddings = np.load("embeddings.npy")

with open("nomes.txt", "r", encoding="utf-8") as f:
    nomes = [linha.strip() for linha in f]

imagem_usuario = preprocess(
    Image.open(sys.argv[1]).convert("RGB")
).unsqueeze(0)

with torch.no_grad():

    feat = model.encode_image(imagem_usuario)
    feat /= feat.norm(dim=-1, keepdim=True)

query = feat.cpu().numpy()[0]

similaridades = embeddings @ query

top = np.argsort(similaridades)[::-1][:5]

print("\nTop 5 resultados:\n")

for i in top:

    print(
        f"{nomes[i]} -> {similaridades[i]*100:.2f}%"
    )