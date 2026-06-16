from flask import Flask, render_template, request, send_from_directory
import torch
import open_clip
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
SKINS_FOLDER = "skins"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Carregando modelo...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model.eval()

print("Carregando embeddings...")

embeddings = np.load("embeddings.npy")

with open("nomes.txt", "r", encoding="utf-8") as f:
    nomes = [linha.strip() for linha in f]

print(f"{len(nomes)} skins carregadas.")

# Detecta automaticamente as extensões das imagens
arquivos_skins = {}

for arquivo in os.listdir(SKINS_FOLDER):

    caminho = os.path.join(SKINS_FOLDER, arquivo)

    if not os.path.isfile(caminho):
        continue

    nome = os.path.splitext(arquivo)[0]

    arquivos_skins[nome] = arquivo


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


@app.route("/skins/<path:filename>")
def skin_file(filename):

    return send_from_directory(
        SKINS_FOLDER,
        filename
    )


@app.route("/", methods=["GET", "POST"])
def index():

    resultados = None
    imagem_url = None

    if request.method == "POST":

        if "imagem" not in request.files:

            return render_template(
                "index.html",
                resultados=None,
                imagem_url=None
            )

        arquivo = request.files["imagem"]

        if arquivo.filename == "":

            return render_template(
                "index.html",
                resultados=None,
                imagem_url=None
            )

        extensao = os.path.splitext(
            arquivo.filename
        )[1]

        nome_unico = (
            str(uuid.uuid4()) + extensao
        )

        caminho_upload = os.path.join(
            UPLOAD_FOLDER,
            nome_unico
        )

        arquivo.save(caminho_upload)

        imagem_url = (
            f"/uploads/{nome_unico}"
        )

        imagem = preprocess(
            Image.open(caminho_upload).convert("RGB")
        ).unsqueeze(0)

        with torch.no_grad():

            feat = model.encode_image(imagem)

            feat /= feat.norm(
                dim=-1,
                keepdim=True
            )

        query = feat.cpu().numpy()[0]

        similaridades = embeddings @ query

        top = np.argsort(
            similaridades
        )[::-1][:5]

        resultados = []

        for i in top:

            nome_skin = nomes[i]

            resultados.append({
                "nome": nome_skin,
                "arquivo": arquivos_skins.get(
                    nome_skin,
                    ""
                ),
                "score": round(
                    float(similaridades[i] * 100),
                    2
                )
            })

    return render_template(
        "index.html",
        resultados=resultados,
        imagem_url=imagem_url
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )