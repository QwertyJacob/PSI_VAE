# VAE for generating faces from CelebA dataset

This is a simple program to train a Variational Autoencoder (VAE) to generate faces from the CelebA dataset.

## Data

The CelebA dataset is a large-scale face attributes dataset with more than 200,000 celebrity images, each with 40 attribute annotations. The dataset can be downloaded from the [CelebA website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
But I reccomend using [kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data) 
Put it into the data folder.
## Model

The model is a Variational Autoencoder (VAE) with a convolutional encoder and a convolutional decoder. The encoder maps the input image to a lower-dimensional latent space, and the decoder maps the latent space back to the original image space.

## Training

The model is trained using the Adam optimizer with a learning rate specified in the conf file. The loss function is the mean squared error (MSE) between the input and reconstructed images.

## Evaluation

The model is evaluated using the Frechet Inception Distance (FID) metric, which measures the similarity between the generated and real images.

## Results

The model is able to generate realistic faces that are similar to the real images in the CelebA dataset. 
The SOTA FID score is 23.12, can we reach them?

## Usage

To use the model, you can run the following command in the terminal:

```bash
# Install required packages
pip install -r requirements.txt
```
```bash
# Train your model (ajdust params first in the conf/default.yaml file)
python3 train.py
```


## More

\documentclass[tikz,border=3.14mm]{standalone}
\usepackage{amsmath}

\begin{document}

\begin{tikzpicture}[
    latent/.style={circle, draw, minimum size=1cm, inner sep=0, node distance=1.5cm},
    observed/.style={rectangle, draw, minimum size=1cm, inner sep=0, node distance=1.5cm},
    arrow/.style={-latex, thick},
    every node/.style={align=center}
]

% Nodes
\node[latent] (z) {$z$};
\node[observed, right=of z] (x) {$x$};
\node[below=1cm of z] (pz) {$p(z)$};
\node[below=1cm of x] (pxz) {$p(x|z)$};
\node[latent, left=of z] (qzx) {$q(z|x)$};

% Edges
\draw[arrow] (pz) -- (z) node[midway, left] {Prior};
\draw[arrow] (z) -- (x) node[midway, above] {Decoder};
\draw[arrow] (z) -- (pxz) node[midway, right] {};
\draw[arrow] (qzx) -- (z) node[midway, above] {Encoder};
\draw[arrow] (qzx) -- (x) node[midway, below] {};

% Labels
\node[below=0.5cm of pxz, align=center] {Likelihood\\$p(x|z)$};
\node[below=0.5cm of pz, align=center] {Prior\\$p(z)$};
\node[below=0.5cm of qzx, align=center] {Posterior Approximation\\$q(z|x)$};

\end{tikzpicture}

\end{document}
