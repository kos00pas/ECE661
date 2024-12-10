# Generative Adversarial Networks 

discriminative models: 
: conditional probability (I/O)
: focus in classification 
: can not generate new data

| **Aspect**               | **Discriminative Models**                                                                            | **Generative Models**                                                                              |
|---------------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **Definition**            | Learn the decision boundary between classes.                                                         | Learn the underlying data distribution.                                                            |
| **Focus**                | Estimate \( P(y\|x) \): Conditional probability.                                                     | Estimate \( P(x) \) or \( P(x\|y) \): Joint probability.                                           |
| **Goal**                 | Predict the label/class for given input.                                                             | Generate new data samples or understand data structure.                                            |
| **Examples**             | Logistic Regression, SVM, Neural Networks.                                                           | GANs, Variational Autoencoders (VAE), Naive Bayes.                                                 |
| **Training Data**        | Requires labeled data for supervised learning.                                                       | Can work with unlabeled data for unsupervised learning.                                            |
| **Strengths**            | - Directly optimize for classification or regression. <br>- Easier to implement and faster to train. | - Can generate realistic new data samples. <br>- Better for understanding the full data structure. |
| **Weaknesses**           | - Cannot generate new data. <br>- Limited understanding of data structure.                           | - Complex to train. <br>- Computationally expensive.                                               |
| **Applications**         | - Classification tasks (e.g., image recognition). <br>- Regression problems.                         | - Image generation (e.g., GANs). <br>- Anomaly detection.                                          |


Generative models 
: aim to understand and learn the underlying distribution of a dataset
: to generate new data samples similar to the original dat


| **Aspect**              | **Generative Models**                                                      | **Deep Generative Models**                                                                          |
|--------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Definition**           | Learn \( P(x) \) or \( P(x\|y) \) using traditional statistical methods.   | Use deep neural networks to learn \( P(x) \) or \( P(x\|y) \).                                      |
| **Complexity**           | Simpler and based on traditional approaches (e.g., Naive Bayes).           | More complex, leveraging deep learning architectures.                                               |
| **Examples**             | Naive Bayes, Hidden Markov Models (HMMs), Gaussian Mixture Models (GMMs).  | Variational Autoencoders (VAEs), GANs, Diffusion Models.                                            |
| **Scalability**          | Limited scalability for high-dimensional data.                             | Highly scalable, especially for large, high-dimensional datasets.                                   |
| **Training Requirements**| Often rely on assumptions about data distributions.                        | Use flexible deep architectures, requiring extensive data and computational resources.              |
| **Expressiveness**       | Limited due to reliance on predefined statistical models.                  | Highly expressive, capable of learning complex data distributions.                                  |
| **Generative Quality**   | Can generate basic data samples but struggle with realistic outputs.       | Generate highly realistic and diverse data samples.                                                 |
| **Applications**         | - Text classification (e.g., Naive Bayes). <br> - Simple clustering tasks. | - Image generation (GANs, VAEs). <br> - Audio synthesis. <br> - Text generation (e.g., GPT models). |
| **Weaknesses**           | - Limited flexibility. <br> - Poor performance on complex data.            | - Computationally intensive. <br> - Harder to train (e.g., mode collapse in GANs).                  |


most representative generative models:
: **Deep belief networks** :  learn hierarchical, probabilistic representations of the input data
: **variational autoencoders** : data generation, latent space learning, and unsupervised feature extraction . use autoencoder
: **Generative adversarial networks** : two NN ( generative/discriminative) . learn and generate realistic data.

Restricted Boltzmann Machine (RBM)
:  energy-based, probabilistic model used for unsupervised learning.
:  learn the probability distribution of input data and extract meaningful features


deep beliefs  networks
: probabilistic  generative model
:  learn hierarchical representations of data and is useful for both unsupervised pre-training and supervised learning

 
autoencoder
: unsupervised learning , learn efficient (encoding =) representation  of input 
: designed to compress and reconstruct data, making it useful for dimensionality reduction, feature extraction, and anomaly detection.
: pair of Encoder-decoder 
: encoder = Compresses the input into a lower-dimensional representation
: decoder =  Reconstructs the input data from this compressed representation.
:  Latent Space: The latent space is the bottleneck layer where the input is represented in a compressed form.
: Loss Function (MSE , BCE)  : Mean Squared Error/Binary Cross-Entropy

![img_2.png](img/_img_2.png)

Variational Autoencoder (VAE)
: Common autoencoder Architecture
: Layers: Encoder and decoder with probabilistic layers (novelty).
: Use: Extends autoencoders to generate new data by sampling from a latent space
: aim to minimize recostruction error  

VAE advantages 
: smooth latent space 
: generative capabilities 
: regularization : latent space, preventing ovverfitting 

VAE disadvantages
: blurry outputs
: complexity 
: expressiveness 


VAE loss 
: Variational Autoencoders (VAEs) optimize
a loss function combining KL divergence,
to align q(z∣X) with p(z) and reconstruction likelihood p(X∣z).
: The reparameterization trick reformulates z 
using μ(X) and σ(X) to enable gradient-based optimization.

KL Divergence 
: KL Divergence is a measure of how 
one probability distribution Q (the approximation) 
differs from another probability distribution P (the true distribution)

Latent Space in Autoencoders
:  the compressed representation or bottleneck layer where the Encoder maps the input data.





---


GAN: Generative adversarial networks 
: Generator (G): Learns to create realistic data (e.g., images) from random noise.
: Discriminator (D): Learns to distinguish between real data (from the dataset) and fake data (produced by the generator).

![img_5.png](img/_img_5.png)

Latent Space in GANs
: a low-dimensional space from which the Generator produces high-dimensional, realistic data.
: Acts as a compressed representation that the Generator learns to map to realistic outputs 

| **Issue**                | **Cause**                                  | **Solution**                           |
|--------------------------|-------------------------------------------|----------------------------------------|
| Training Instability     | Imbalance between Generator and Discriminator | Use WGAN or improved optimizers       |
| Mode Collapse            | Generator focuses on limited outputs       | Mini-batch discrimination, WGAN       |
| Vanishing Gradients      | Discriminator becomes too strong/weak      | Gradient clipping, normalization       |
| Overfitting in Discriminator | Memorization of training data            | Data augmentation, dropout             |
| Evaluation Difficulty    | No universal metric                        | Use IS or FID as approximate measures  |
| High Computational Cost  | Complex network training                   | Use smaller architectures              |
| Theoretical Challenges   | Complex adversarial dynamics               | Trial-and-error, systematic research   |
| Hyperparameter Sensitivity | Poor performance without tuning           | Grid search, best practices            |


The Generator (G) and Discriminator (D) play a zero-sum game:
: A zero-sum game is a concept from game theory where the gain of one player equals the loss of the other player, meaning the total payoff is always zero.

![img_3.png](img/_img_3.png)


GAN Training Procedure
:  Simultaneous Training G&D with SGD
: Minibatches include:
Generated samples (from the Generator) & 
Real-world samples (from the dataset).
: Training G :Error for G comes via backpropagation through D
: also can be Independent Training
: Feedback Loop:
G improves by generating samples that better fool D.
D improves by distinguishing real samples from those generated by G.

loss function :
: ![img_4.png](img/_img_4.png)




























































































