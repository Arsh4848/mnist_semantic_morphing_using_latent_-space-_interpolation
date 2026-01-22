import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from torchvision import datasets, transforms
from PIL import Image


# CONFIG
LATENT_DIM = 20
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
MODEL_PATH = "vae_mnist.pth"
DEVICE = "cpu"

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# FLASK

app = Flask(__name__)

# DATA

transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# VAE MODEL
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
        recon = self.decode(z)
        return recon, mu, logvar


# LOSS

def vae_loss(recon, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(
        recon, x.view(-1, 784), reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


# LOAD / TRAIN
vae = VAE().to(DEVICE)

if os.path.exists(MODEL_PATH):
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    vae.train()
    for _ in range(EPOCHS):
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
    torch.save(vae.state_dict(), MODEL_PATH)

vae.eval()

# HELPERS
def load_img(path):
    img = Image.open(path).convert("L").resize((28, 28))
    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img.to(DEVICE)

def morph(img1, img2, steps):
    with torch.no_grad():
        z1, _ = vae.encode(img1)
        z2, _ = vae.encode(img2)

        frames = []
        for a in np.linspace(0, 1, steps):
            z = (1 - a) * z1 + a * z2
            img = vae.decode(z).view(28, 28).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.resize(img, (256, 256))
            frames.append(img)
        return frames

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    steps = int(request.form["steps"])
    mode = request.form["mode"]

    if mode == "single":
        s1 = load_img(request.files["src1"])
        t1 = load_img(request.files["tgt1"])
        frames = morph(s1, t1, steps)
    # double digit
    else: 
        s1 = load_img(request.files["src1"])
        s2 = load_img(request.files["src2"])
        t1 = load_img(request.files["tgt1"])
        t2 = load_img(request.files["tgt2"])

        frames1 = morph(s1, t1, steps)
        frames2 = morph(s2, t2, steps)

        frames = [
            np.hstack((f1, f2))
            for f1, f2 in zip(frames1, frames2)
        ]

    path = os.path.join(OUTPUT_FOLDER, "morph.avi")
    h, w = frames[0].shape
    video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"XVID"),15,(w, h),False)

    for f in frames:
        video.write(f)
    video.release()

    return send_from_directory(OUTPUT_FOLDER, "morph.avi")


# RUN
if __name__ == "__main__":
    app.run(debug=True)
