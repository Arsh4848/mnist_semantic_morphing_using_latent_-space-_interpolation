import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import cv2
import random
import os

# =============================
# CONFIG
# =============================
#for saving frames 
FRAMES_DIR = "morph_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

LATENT_DIM = 20
BATCH_SIZE = 128
EPOCHS = 20                 # good quality, train once
LR = 1e-3
STEPS = 40
MODEL_PATH = "vae_mnist.pth"
VIDEO_NAME = "mnist_latent_morphing.avi"
DEVICE = torch.device("cpu")

# =============================
# DATA
# =============================
transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

# =============================
# VAE MODEL
# =============================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.mu = nn.Linear(400, LATENT_DIM)
        self.logvar = nn.Linear(400, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# =============================
# LOSS
# =============================
def vae_loss(recon, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(
        recon, x.view(-1, 784), reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

# =============================
# TRAIN OR LOAD MODEL
# =============================
vae = VAE().to(DEVICE)

if os.path.exists(MODEL_PATH):
    print("✅ Loading pre-trained VAE model...")
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("🚀 Training VAE (ONE TIME)...")
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    vae.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader.dataset):.4f}")

    torch.save(vae.state_dict(), MODEL_PATH)
    print("💾 Model saved as vae_mnist.pth")

vae.eval()

# =============================
# USER LOOP (ANYTIME INPUT)
# =============================


##Allow user to input or upload images locally from device 
def load_user_image(path):
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid path")
    
    img = 255 - img
 
    img = cv2.resize(img, (28, 28))
  
    img = img.astype(np.float32) / 255.0
 
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    return img.to(DEVICE)

while True:
    src = input("\nEnter source digit (0-9) or 'q' to quit: ")
    if src.lower() == 'q':
        break

    tgt = input("Enter target digit (0-9): ")

    #src_digit = int(src)
    #tgt_digit = int(tgt)

    # pick random MNIST test samples
    #src_indices = [i for i, (_, label) in enumerate(test_dataset) if label == src_digit]
    #tgt_indices = [i for i, (_, label) in enumerate(test_dataset) if label == tgt_digit]

    #img1, _ = test_dataset[random.choice(src_indices)]
    #img2, _ = test_dataset[random.choice(tgt_indices)]

    #img1 = img1.unsqueeze(0).to(DEVICE)
    #img2 = img2.unsqueeze(0).to(DEVICE)
    img1 = load_user_image(src)
    img2 = load_user_image(tgt)

    print(f"🎥 Generating morphing video:")

    frames = []
    with torch.no_grad():
        z1, _ = vae.encode(img1)
        z2, _ = vae.encode(img2)

        """for alpha in np.linspace(0, 1, STEPS):
            z = (1 - alpha) * z1 + alpha * z2
            decoded = vae.decode(z).view(28, 28).cpu().numpy()
            frame = (decoded * 255).astype(np.uint8)
            frame = cv2.resize(frame, (256, 256))
            frames.append(frame)"""
        for i, alpha in enumerate(np.linspace(0, 1, STEPS)):
            z = (1 - alpha) * z1 + alpha * z2
            decoded = vae.decode(z).view(28, 28).cpu().numpy()

            frame = (decoded * 255).astype(np.uint8)
            frame = cv2.resize(frame, (256, 256))

            # save frame as JPG
            frame_path = os.path.join(FRAMES_DIR, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)

            frames.append(frame)
    

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(VIDEO_NAME, fourcc, 15, (256, 256))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    video.release()
    print(f"✅ Saved: {VIDEO_NAME}")
