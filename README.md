In this project  we are demonstrating semantic morphing of handwritten digits using a Variational Autoencoder (VAE) trained on the MNIST dataset.
What is the MNIST Dataset?
The MNIST dataset is a widely used benchmark dataset in machine learning and computer vision.

Given any two handwritten digits from the MNIST test dataset (for example 3 → 8), the model generates a smooth transformation video by interpolating between their latent representations.
The transformation is performed in latent space, not pixel space, which results in meaningful and visually smooth transitions.

 I have used VAE(variational autoencoder) technique in this project to develop the model and OpenCV is used to create video of frames during digit transformation 

A Variational Autoencoder (VAE) is a type of generative deep learning model used to learn a compact and continuous representation of data.

      Structure of a VAE
            
A VAE consists of two main parts:
Encoder : Takes an input image  ,Compresses it into a latent representation .Outputs two vectors ,Mean (μ)Log variance (log σ²)
Decoder:Takes a sampled latent vector,Reconstructs the original image
Unlike a normal autoencoder, a VAE learns a probabilistic latent space, typically following a Gaussian distribution.

Why VAE is Used in This Project
The goal of this project is to smoothly transform one handwritten digit into another.
Why not pixel interpolation?
Pixel-wise interpolation: Produces blurry images and also Does not preserve semantic meaning
Why VAE works well
A VAE:Learns a continuous and smooth latent space Ensures nearby latent vectors produce similar images Allows meaningful interpolation between two digits This makes VAEs ideal for semantic morphing and latent space interpolation tasks.
 Latent Space Interpolation (Core Idea)
After encoding two digits into latent vectors z₁ and z₂, intermediate latent vectors are generated using linear interpolation:
z(α) = (1 − α) · z₁ + α · z₂


Where:
α varies from 0 to 1
z₁ represents the source digit
z₂ represents the target digit
Each interpolated latent vector is decoded back into an image, producing a smooth transition between digits.


In this project:

The training set is used to train the VAE and The test set is used to select input digits dynamically

How OpenCV is Used to Create the Video
OpenCV is a powerful computer vision library that supports image and video processing.

Role of OpenCV in This Project
OpenCV is used to Convert generated images into frames
Resize frames for better visibility
Write frames sequentially into a video file
Video Generation Process Each decoded image is converted to a NumPy array
Pixel values are scaled to 0–255
Frames are written using cv2.VideoWriter

Output format: .avi (widely supported)


Key Technologies Used

Python
PyTorch – deep learning framework
Torchvision – MNIST dataset handling
NumPy – numerical operations
OpenCV – video generation


In the end i have also attached the example video of converting 9 ->3
