import os
import math

from matplotlib.pylab import cond
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 128
TIMESTEPS = 1000
BATCH_SIZE = 48
EPOCHS = 100
LR = 1e-4
DDIM_STEPS = 100
step_size = TIMESTEPS // DDIM_STEPS

DROP_PROB = 0.2  # classifier-free guidance

ORIGINAL_PATH = "/home/teaching/dl27/data/Resized_D1/Eye_Disease_Image_Dataset/Original_Dataset/Original_Dataset"
AUGMENTED_PATH = "/home/teaching/dl27/data/Resized_D1/Eye_Disease_Image_Dataset/Augmented_Dataset/Augmented_Data"

AE_PATH = "../latent_autoencoder/autoencoder_results/best_ae.pt"

SAVE_DIR = "./new_net_output_3"
os.makedirs(SAVE_DIR, exist_ok=True)

latest_model = f"{SAVE_DIR}/model_latest.pt"
latest_ema   = f"{SAVE_DIR}/ema_latest.pt"
best_model   = f"{SAVE_DIR}/best_model.pt"
best_ema     = f"{SAVE_DIR}/best_ema.pt"
# -------------------------
# DATA
# -------------------------
import cv2
import numpy as np
from PIL import Image

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

#         img = np.stack([green, green, green], axis=-1)

#         return Image.fromarray(img)
    
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset1 = datasets.ImageFolder(ORIGINAL_PATH, transform=transform)
dataset2 = datasets.ImageFolder(AUGMENTED_PATH, transform=transform)

dataset = ConcatDataset([dataset1, dataset2])
# weighed loader to handle class imbalance
from torch.utils.data import WeightedRandomSampler

targets = [y for _, y in dataset]
class_counts = torch.bincount(torch.tensor(targets))
weights = 1.0 / class_counts.float()
sample_weights = [weights[t] for t in targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
##
num_classes = len(dataset1.classes)

print("Classes:", dataset1.classes)
print("Total:", len(dataset))

# -------------------------
# AUTOENCODER
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
ckpt = torch.load(AE_PATH, map_location=DEVICE)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

decoder = Decoder().to(DEVICE)
decoder.load_state_dict(ckpt["decoder"])
decoder.eval()

for p in decoder.parameters():
    p.requires_grad = False

# -------------------------
# DIFFUSION SCHEDULE
# -------------------------
def cosine_beta_schedule(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas = torch.cos((x/timesteps)*math.pi*0.5)**2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas,1e-4,0.999)

betas = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas,0)

# -------------------------
# MODEL
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super().__init__()

        g1 = min(groups, in_c)
        g2 = min(groups, out_c)
        

        self.block = nn.Sequential(
            nn.GroupNorm(g1, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3,1,1),

            nn.GroupNorm(g2, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3,1,1)
        )

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self,x):
        return self.block(x) + self.skip(x)
    
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(in_c, out_c),
            ResBlock(out_c, out_c)
        )
        self.down = nn.Conv2d(out_c, out_c, 4, 2, 1)  # ↓ spatial

    def forward(self,x):
        x = self.block(x)
        return x, self.down(x)
    
class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.block = nn.Sequential(
         ResBlock(in_c + out_c, out_c),
         ResBlock(out_c, out_c)
            )

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class BetterUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes + 1, 256)
        self.cond_proj1 = nn.Conv2d(256, 128, 1)
        self.cond_proj2 = nn.Conv2d(256, 128, 1)
        self.cond_proj3 = nn.Conv2d(256, 256, 1)

        self.time_mlp = nn.Sequential(
            nn.Linear(1,256),
            nn.SiLU(),
            nn.Linear(256,256)
        )

        # initial projection
        self.in_conv = nn.Conv2d(4, 128, 3,1,1)

        # down
        self.down1 = Down(128,128)   # 32 → 16
        self.down2 = Down(128,256)   # 16 → 8

        # middle
        self.mid = nn.Sequential(
            ResBlock(256,256),
            ResBlock(256,256)
        )

        # up
        self.up1 = Up(256,128)       # 8 → 16
        self.up2 = Up(128,128)       # 16 → 32

        # output
        self.out = nn.Conv2d(128,4,3,1,1)

    def forward(self,x,t,y):

        # conditioning
        t = t.float().unsqueeze(-1)/TIMESTEPS
        t = self.time_mlp(t)[:,:,None,None]

        y = self.label_emb(y)[:,:,None,None]

        cond = t + y
        cond = cond*2.0
        cond1 = self.cond_proj1(cond)  # for 128
        cond2 = self.cond_proj2(cond)  # for 128
        cond3 = self.cond_proj3(cond)  # for 256

        # input
        x = self.in_conv(x)

        # down
        x = x + cond1
        s1, x = self.down1(x)
        x = x + cond2
        s2, x = self.down2(x)

        # mid
        x = x + cond3
        x = self.mid(x)

        # up
        x = self.up1(x, s2)
        x = self.up2(x, s1)

        return self.out(x)

model = BetterUNet(num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# EMA
# -------------------------
ema_model = deepcopy(model)
EMA_DECAY = 0.9995

# -------------------------
# RESUME
# -------------------------
START_EPOCH = 0

latest_model = f"{SAVE_DIR}/model_latest.pt"
latest_ema = f"{SAVE_DIR}/ema_latest.pt"

if os.path.exists(best_model):
    print("Loading BEST model for fine-tuning...")
    model.load_state_dict(torch.load(best_model, map_location=DEVICE))
    ema_model.load_state_dict(torch.load(best_ema, map_location=DEVICE))
    START_EPOCH = 0   # restart epoch count for fine-tuning

## SAMPLER ##
@torch.no_grad()
def sample(model, n, label):

    x = torch.randn(n,4,32,32).to(DEVICE)
    labels = torch.full((n,), label, device=DEVICE)

    for i in reversed(range(0,TIMESTEPS,step_size)):
        t = torch.full((n,), i, device=DEVICE)

        # conditional
        pred_cond = model(x, t, labels)

        # unconditional (label = num_classes)
        uncond_labels = torch.full_like(labels, num_classes)
        pred_uncond = model(x, t, uncond_labels)

        # guidance
        GUIDANCE_SCALE = 3.0
        pred = pred_uncond + GUIDANCE_SCALE * (pred_cond - pred_uncond)

        alpha = alphas_cumprod[t][:,None,None,None]

        x0 = (x - torch.sqrt(1-alpha)*pred) / torch.sqrt(alpha)

        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        prev_t = torch.clamp(t - step_size, min=0)
        alpha_prev = alphas_cumprod[prev_t][:,None,None,None]

        x = torch.sqrt(alpha_prev)*x0 + torch.sqrt(1-alpha_prev)*noise

    return x
# -------------------------
# TRAIN
# -------------------------
best_loss = float("inf")

for epoch in range(START_EPOCH, EPOCHS):

    pbar = tqdm(loader)

    for imgs, labels in pbar:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        # classifier-free dropout
        mask = torch.rand(labels.shape, device=DEVICE) < DROP_PROB
        labels = torch.where(mask, torch.full_like(labels, num_classes), labels)

        with torch.no_grad():
            z = encoder(imgs)
            mean = z.mean(dim=[1,2,3], keepdim=True)
            std  = z.std(dim=[1,2,3], keepdim=True)
            z = (z - mean) / (std + 1e-6)

        t = torch.randint(0, TIMESTEPS, (z.size(0),), device=DEVICE)
        noise = torch.randn_like(z)

        alpha = alphas_cumprod[t][:,None,None,None]
        z_noisy = torch.sqrt(alpha)*z + torch.sqrt(1-alpha)*noise

        pred_noise = model(z_noisy, t, labels)

        loss = ((noise - pred_noise)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)

        pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

    # --- SAVE LATEST ---
    torch.save(model.state_dict(), latest_model)
    torch.save(ema_model.state_dict(), latest_ema)
    torch.save(epoch, f"{SAVE_DIR}/epoch.pt")

    # --- SAVE BEST ---
    if loss.item() < best_loss:
        best_loss = loss.item()

        torch.save(model.state_dict(), best_model)
        torch.save(ema_model.state_dict(), best_ema)

        print(f"Saved BEST model at epoch {epoch} (loss={best_loss:.4f})")

    #SAMPLE 
    if epoch % 50 == 0:
        print(f"Sampling at epoch {epoch}...")

        model_to_use = ema_model

        all_imgs = []

        for cls in range(num_classes):

            z = sample(model_to_use, 8, label=cls)

            imgs = decoder(z)
            imgs = torch.clamp(imgs, -1, 1)

            all_imgs.append(imgs)

        imgs = torch.cat(all_imgs, dim=0)

        from torchvision.utils import save_image
        save_image((imgs+1)/2, f"{SAVE_DIR}/sample_{epoch}.png", nrow=8)

    print(f"Saved epoch {epoch}")