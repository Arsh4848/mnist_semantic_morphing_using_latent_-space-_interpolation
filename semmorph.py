import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from torchvision import datasets, transforms
from PIL import Image

# =============================
# CONFIG
# =============================
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


# FLASK APP

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# =============================
# DATA
# =============================
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
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(
        recon, x.view(-1, 784), reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld



# LOAD / TRAIN MODEL (ONCE)

vae = VAE().to(DEVICE)

if os.path.exists(MODEL_PATH):
    print("Loading trained VAE...")
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("Training VAE...")
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

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_loader.dataset):.4f}")

    torch.save(vae.state_dict(), MODEL_PATH)
    print("Model saved")

vae.eval()


# IMAGE PREPROCESS

def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img.to(DEVICE)


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    source = request.files["source"]
    target = request.files["target"]
    steps = int(request.form["steps"])

    src_path = os.path.join(UPLOAD_FOLDER, "src.png")
    tgt_path = os.path.join(UPLOAD_FOLDER, "tgt.png")

    source.save(src_path)
    target.save(tgt_path)

    img1 = load_image(src_path)
    img2 = load_image(tgt_path)

    with torch.no_grad():
        mu1, _ = vae.encode(img1)
        mu2, _ = vae.encode(img2)

        frames = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            decoded = vae.decode(z).view(28, 28).cpu().numpy()
            frame = (decoded * 255).astype(np.uint8)
            frame = cv2.resize(frame, (256, 256))
            frames.append(frame)

    video_path = os.path.join(OUTPUT_FOLDER, "morph.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_path, fourcc, 15, (256, 256))

    for f in frames:
        video.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))

    video.release()

    return send_from_directory(OUTPUT_FOLDER, "morph.avi")



# RUN

if __name__ == "__main__":
    app.run(debug=True)
