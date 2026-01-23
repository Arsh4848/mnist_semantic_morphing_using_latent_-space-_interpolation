Semantic Morphing of Handwritten Digits using VAE (MNIST)
This project demonstrates semantic morphing of handwritten digits using a Variational Autoencoder (VAE) trained on the MNIST dataset. It supports both single-digit and double-digit transformations and generates a smooth transition video between source and target digits using latent space interpolation.
KEY FEATURES
Variational Autoencoder (VAE) trained on MNIST
Latent space semantic interpolation
Supports single-digit morphing (e.g., 7 → 2)
Supports double-digit morphing (e.g., 32 → 41)
User-defined number of intermediate steps
Flask-based web application
Video generation using OpenCV
Model is trained once and reused anytime
PROBLEM STATEMENT
Input:
Image A (Source): Grayscale handwritten digit image(s)
Image B (Target): Grayscale handwritten digit image(s)
N (Steps): Number of intermediate transition frames
Output:
A video showing smooth semantic transformation from Image A to Image B
WHY VARIATIONAL AUTOENCODER (VAE)?
A Variational Autoencoder learns a continuous and structured latent space where
similar digits are located close to each other. Linear interpolation between
latent vectors produces smooth and meaningful visual transitions, making VAE
suitable for semantic morphing problems.
WHY we avoided GANs ?
GANs will be difficult to interpolate the output image and it is also hard to train
DATASET: MNIST
70,000 grayscale images
Image size: 28 × 28
Digits: 0 to 9
Widely used benchmark dataset for handwritten digit recognition and
generative modeling
PROJECT ARCHITECTURE:
Training Phase:
MNIST Dataset
↓
Encoder Network
(784 → 400 → μ, σ)
↓
Reparameterization Trick
↓
Latent Space (20-dimensional)
↓
Decoder Network
(20 → 400 → 784)
↓
Reconstructed Image
CONTROL FLOW (END-TO-END)
Model Training (One Time):
Load MNIST training data
↓
Train VAE using reconstruction loss and KL divergence
↓
Save trained model to disk
User Interaction (Flask Web App):
↓
User uploads source and target images
↓
User specifies number of steps
↓
Images are resized and normalized
↓
Images are encoded into latent vectors
↓
Latent vectors are interpolated linearly
↓
Each latent vector is decoded into an image frame
↓
Frames are combined (single or double digit)
↓
OpenCV generates a morphing video
↓
Video is returned to the user
SINGLE DIGIT VS DOUBLE DIGIT LOGIC
Single Digit (e.g., 7 → 2):
One source image
One target image
Direct latent space interpolation
Double Digit (e.g., 32 → 41):
Each digit is treated independently

  3   2
  ↓   ↓
  4   1


Morph first digit (3 → 4)
Morph second digit (2 → 1)

Frames are concatenated side-by-side
This preserves semantic correctness without retraining

LATENT SPACE INTERPOLATION
Let:
z1 = Encoder(Image A)
z2 = Encoder(Image B)
Interpolation:
z(alpha) = (1 − alpha) * z1 + alpha * z2
where alpha ranges from 0 to 1
Each interpolated latent vector is decoded to form a transition frame.
LOSS FUNCTION USED IN VAE (used for backpropagation to modify the weight and biases in network)
Reconstruction Loss:
Binary Cross Entropy (BCE)
Ensures decoded image matches the input image
KL Divergence Loss:
Regularizes latent space to follow a normal distribution(gaussian distribution)
Total Loss:
Loss = Reconstruction Loss + KL Divergence Loss
VIDEO GENERATION
Each decoded frame is resized for better visualization
Frames are written sequentially into a video file
OpenCV's VideoWriter is used to generate an .AVI video
TECHNOLOGIES USED
PyTorch: Model training and inference
MNIST Dataset: Handwritten digits
Flask: Web application framework
OpenCV: Video generation
PIL / NumPy: Image preprocessing
HTML: Frontend interface
PROJECT STRUCTURE
project/
│
├── app.py
├── vae_mnist.pth
│
├── templates/
│   └── index.html
│
├── static/
│   ├── uploads/
│   └── outputs/
HOW TO RUN
Install dependencies:
pip install torch torchvision flask opencv-python pillow numpy
Run the application:
python app.py
Open browser and visit:
http://127.0.0.1:5000/
LEARNING OUTCOMES
Understanding Variational Autoencoders
Latent space representation and interpolation
Semantic image morphing
Integration of machine learning with Flask
Video generation using OpenCV
