import os
import torch
import torch.nn as nn
from PIL import Image
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo: {DEVICE}")

# ==========================
# CARREGA CLIP
# ==========================

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model = model.to(DEVICE)

# ==========================
# CLASSES
# ==========================

arquivos = sorted(os.listdir("skins"))

classes = []

for arquivo in arquivos:

    caminho = os.path.join(
        "skins",
        arquivo
    )

    if os.path.isfile(caminho):

        classes.append(
            os.path.splitext(arquivo)[0]
        )

num_classes = len(classes)

print(
    f"{num_classes} classes carregadas."
)

# ==========================
# CLASSIFICADOR
# ==========================

classifier = nn.Linear(
    model.visual.output_dim,
    num_classes
).to(DEVICE)

# ==========================
# CARREGA PESOS
# ==========================

checkpoint = torch.load(
    "skin_clip_finetuned.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["clip"]
)

classifier.load_state_dict(
    checkpoint["classifier"]
)

model.eval()
classifier.eval()

print("Modelo carregado.")

# ==========================
# IMAGEM
# ==========================

imagem_teste = "skins/AK-47 Asiimov.png"

imagem = preprocess(
    Image.open(imagem_teste).convert("RGB")
).unsqueeze(0)

imagem = imagem.to(DEVICE)

# ==========================
# INFERÊNCIA
# ==========================

with torch.no_grad():

    features = model.encode_image(
        imagem
    )

    logits = classifier(
        features
    )

    probs = torch.softmax(
        logits,
        dim=1
    )

    top_probs, top_idx = torch.topk(
        probs,
        k=5
    )

print("\nTop 5 resultados:\n")

for prob, idx in zip(
    top_probs[0],
    top_idx[0]
):

    nome_skin = classes[idx.item()]

    print(
        f"{nome_skin} -> "
        f"{prob.item()*100:.2f}%"
    )