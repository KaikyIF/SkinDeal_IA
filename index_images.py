import torch
import open_clip
import numpy as np
from PIL import Image
from pathlib import Path

print("Carregando modelo...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model.eval()

embeddings = []
nomes = []

skins_dir = Path("skins")

for arquivo in skins_dir.glob("*"):

    if arquivo.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
        continue

    print(f"Processando: {arquivo.name}")

    imagem = preprocess(
        Image.open(arquivo).convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():

        feat = model.encode_image(imagem)
        feat /= feat.norm(dim=-1, keepdim=True)

    embeddings.append(
        feat.cpu().numpy()[0]
    )

    nomes.append(
        arquivo.stem
    )

np.save("embeddings.npy", np.array(embeddings))

with open("nomes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(nomes))

print(f"{len(nomes)} skins indexadas!")