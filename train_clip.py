import os
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

import open_clip

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo: {DEVICE}")

# ==================================
# AUGMENTATION
# ==================================

augmentation = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    )
])

# ==================================
# OPENCLIP
# ==================================

print("Carregando OpenCLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model = model.to(DEVICE)

# Congela tudo

for param in model.parameters():
    param.requires_grad = False

# Libera apenas encoder visual

for param in model.visual.parameters():
    param.requires_grad = True

# ==================================
# DATASET
# ==================================

class SkinDataset(Dataset):

    def __init__(self, pasta):

        self.imagens = []

        arquivos = sorted(
            os.listdir(pasta)
        )

        for arquivo in arquivos:

            caminho = os.path.join(
                pasta,
                arquivo
            )

            if os.path.isfile(caminho):

                self.imagens.append(
                    caminho
                )

        self.labels = list(
            range(
                len(self.imagens)
            )
        )

    def __len__(self):

        return len(
            self.imagens
        )

    def __getitem__(self, idx):

        imagem = Image.open(
            self.imagens[idx]
        ).convert("RGB")

        imagem = augmentation(
            imagem
        )

        imagem = preprocess(
            imagem
        )

        label = self.labels[idx]

        return imagem, label


dataset = SkinDataset(
    "skins"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

num_classes = len(
    dataset
)

print(
    f"{num_classes} skins carregadas."
)

# ==================================
# CLASSIFICADOR
# ==================================

classifier = nn.Linear(
    model.visual.output_dim,
    num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    list(
        model.visual.parameters()
    ) +
    list(
        classifier.parameters()
    ),
    lr=1e-5
)

# ==================================
# TREINAMENTO
# ==================================

EPOCHS = 20

print("\nIniciando treinamento...\n")

for epoch in range(EPOCHS):

    model.train()
    classifier.train()

    total_loss = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        unit="batch"
    )

    for imagens, labels in pbar:

        imagens = imagens.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        features = model.encode_image(
            imagens
        )

        logits = classifier(
            features
        )

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    media_loss = (
        total_loss /
        len(loader)
    )

    print(
        f"\nEpoch {epoch+1}/{EPOCHS} "
        f"finalizada | "
        f"Loss média: {media_loss:.4f}\n"
    )

# ==================================
# SALVAR
# ==================================

torch.save(
    {
        "clip": model.state_dict(),
        "classifier": classifier.state_dict()
    },
    "skin_clip_finetuned.pt"
)

print("\nTreinamento concluído!")

print(
    "Modelo salvo em:"
)

print(
    "skin_clip_finetuned.pt"
)