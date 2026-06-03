import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, ConcatDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 128
BATCH_SIZE = 48
EPOCHS = 150
LR = 2e-4

ORIGINAL_PATH = "/home/teaching/dl27/data/Resized_D1/Eye_Disease_Image_Dataset/Original_Dataset/Original_Dataset"
AUGMENTED_PATH = "/home/teaching/dl27/data/Resized_D1/Eye_Disease_Image_Dataset/Augmented_Dataset/Augmented_Data"

SAVE_DIR = "./autoencoder_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# PREPROCESSING
# -------------------------
# class FundusPreprocess:
#     def __call__(self, img):
#         img = np.array(img)

#         green = img[:,:,1]

#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         green = clahe.apply(green)

#         kernel = np.array([[0,-1,0],
#                            [-1,5,-1],
#                            [0,-1,0]])
#         green = cv2.filter2D(green, -1, kernel)

#         # keep RGB, enhance green only
#         alpha = 0.15  # strength of enhancement
#         img[:,:,1] = (1 - alpha) * img[:,:,1] + alpha * green
#         img = np.clip(img, 0, 255).astype(np.uint8)

#         return Image.fromarray(img)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# DATA
# -------------------------
dataset1 = datasets.ImageFolder(ORIGINAL_PATH, transform=transform)
dataset2 = datasets.ImageFolder(AUGMENTED_PATH, transform=transform)
dataset = ConcatDataset([dataset1, dataset2])

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

print("Total images:", len(dataset))

# -------------------------
# MODEL
# -------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.GroupNorm(8,64),
            nn.SiLU(),

            nn.Conv2d(64,128,4,2,1),
            nn.GroupNorm(8,128),
            nn.SiLU(),

            nn.Conv2d(128,256,3,1,1),
            nn.GroupNorm(8,256),
            nn.SiLU(),

            nn.Conv2d(256,4,1)
        )

    def forward(self,x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,256,3,1,1),
            nn.SiLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256,128,3,1,1),
            nn.SiLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128,64,3,1,1),
            nn.SiLU(),

            nn.Conv2d(64,3,3,1,1),
            nn.Tanh()
        )

    def forward(self,z):
        return self.net(z)


encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR
)
## mocel load continue training
# checkpoint_path = f"{SAVE_DIR}/best_ae.pt"

# if os.path.exists(checkpoint_path):
#     print("🔄 Loading checkpoint...")
#     ckpt = torch.load(checkpoint_path, map_location=DEVICE)

#     encoder.load_state_dict(ckpt["encoder"])
#     decoder.load_state_dict(ckpt["decoder"])
# -------------------------
# PERCEPTUAL LOSS
# -------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(DEVICE)

    def forward(self, x, y):
        return torch.mean((self.vgg(x) - self.vgg(y))**2)

perceptual_loss = VGGPerceptualLoss()

# -------------------------
# TRAIN
# -------------------------
best_loss = float("inf")

for epoch in range(EPOCHS):

    pbar = tqdm(loader)

    for imgs, _ in pbar:

        imgs = imgs.to(DEVICE)

        z = encoder(imgs)
        # --- latent regularization ---
        z_mean = z.mean(dim=[1,2,3])
        z_std  = z.std(dim=[1,2,3])

        latent_loss = torch.mean((z_mean)**2) + torch.mean((z_std - 1)**2)
        recon = decoder(z)

        # losses
        l1 = torch.mean(torch.abs(recon - imgs))
        perc = perceptual_loss((recon+1)/2, (imgs+1)/2)
        color_loss = torch.mean((recon - imgs)**2, dim=[2,3]).mean()

        loss = ( 0.7 * l1 + 0.1 * perc +0.05 * latent_loss + 0.5 * color_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

    # -------------------------
    # SAVE BEST
    # -------------------------
    if loss.item() < best_loss:
        best_loss = loss.item()

        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict()
        }, f"{SAVE_DIR}/best_ae.pt")

    # periodic save
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict()
    }, f"{SAVE_DIR}/ae_{epoch}.pt")

print("Training complete.")