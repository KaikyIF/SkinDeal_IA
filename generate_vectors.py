import os
import json
import numpy as np
from PIL import Image
import torch
import open_clip

print("Carregando modelo IA...")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='laion2b_s32b_b82k'
)

vectors = []
paths = []

image_folder = "images"

files = os.listdir(image_folder)

print(f"{len(files)} imagens encontradas")

for index, file in enumerate(files):

    if file.endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(image_folder, file)

        try:
            image = Image.open(path).convert("RGB")

            image = image.resize((336, 336))

            image = preprocess(
                Image.open(path).convert("RGB")
            ).unsqueeze(0)

            with torch.no_grad():
                features = model.encode_image(image)

            normalized = features / features.norm(dim=-1, keepdim=True)

            vectors.append(normalized[0].numpy())
            paths.append(file)

            print(f"[{index+1}] Vetorizado: {file}")

        except Exception as e:
            print(f"Erro em {file}: {e}")

np.save("image_vectors.npy", np.array(vectors))

with open("image_paths.json", "w") as f:
    json.dump(paths, f)

print("Vetores salvos com sucesso!")