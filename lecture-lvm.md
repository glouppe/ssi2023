class: middle, center, title-slide

# Deep generative models

51st SLAC Summer Institute

August 15, 2023

<br>

Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

# Outline

- Deep generative models
- Variational inference
- Variational auto-encoders
- Hierarchical variational auto-encoders
- Variational diffusion models

---

count: false
class: middle

# Deep generative models

---

class: middle

.grid[
.kol-1-3[.circle.width-95[![](figures/lec11/lecun.jpg)]]
.kol-2-3[.width-100[![](figures/lec11/cake.png)]]
]

.italic["We need tremendous amount of information to build machines that have common sense and generalize."]

.pull-right[Yann LeCun, 2016.]

---

class: middle

## Generative models

A **generative model** is a probabilistic model $p$ that can be used as a simulator of the data.
Its purpose is to generate synthetic but realistic high-dimensional data
$$\mathbf{x} \sim p\_\theta(\mathbf{x}),$$
that is as close as possible from the unknown data distribution $p(\mathbf{x})$, but for which we have empirical samples.

---

class: black-slide
background-image: url(./figures/landscape.png)
background-size: contain

.footnote[Credits: [Karsten et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

---

class: middle

## Content generation

.center[.width-45[![](./figures/lec12/content-generation-1.png)] .width-45[![](./figures/lec12/content-generation-2.png)]]

.center[Diffusion models have emerged as powerful generative models, beating previous state-of-the-art models (such as GANs) on a variety of tasks.]

.footnote[Credits: [Dhariwal and Nichol](https://arxiv.org/pdf/2105.05233.pdf), 2021; [Ho et al](https://arxiv.org/pdf/2106.15282.pdf), 2021.]

---

class: middle

## Image super-resolution

.center[

<video autoplay muted loop width="720" height="420">
     <source src="./figures/lec12/super-resolution.m4v" type="video/mp4">
</video>

]

.footnote[Credits: [Saharia et al](https://arxiv.org/abs/2104.07636), 2021.]

---

class: middle

## Compression

.center[
.width-90[![](figures/lec11/generative-compression.png)]

Hierarchical .bold[compression of images and other data],<br> e.g., in video conferencing systems (Gregor et al, 2016).
]

---

class: middle

## Text-to-image generation

.center[

.width-50[![](./figures/lec12/text-to-image.png)]

.italic[A group of teddy bears in suite in a corporate office celebrating<br> the birthday of their friend. There is a pizza cake on the desk.]

]

.footnote[Credits: [Saharia et al](https://arxiv.org/abs/2205.11487), 2022.]

---

class: middle, black-slide

.center.width-50[![](./figures/lec12/pope.jpg)]

.center[... or deepfakes.]

---

class: middle

## Artistic tools and image editing

.center.width-100[![](./figures/lec12/sde-edit.jpg)]

.footnote[Credits: [Meng et al](https://arxiv.org/abs/2108.01073), 2021.]

---

class: middle

## Voice conversion

.center[

.width-80[![](figures/lec11/vae-styletransfer.jpg)]

.bold[Voice style transfer] [[demo](https://avdnoord.github.io/homepage/vqvae/)] (van den Oord et al, 2017).
]

---

class: middle

## Inverse problems in medical imaging

.center.width-100[![](./figures/lec12/inverse-problems.png)]

.footnote[Credits: [Song et al](https://arxiv.org/pdf/2111.08005.pdf), 2021.]

---

class: middle

## Drug discovery

.center.width-100[![](figures/lec11/bombarelli.jpeg)]

.center[Design of new molecules with desired chemical properties<br> (Gomez-Bombarelli et al, 2016).]

---

count: false
class: middle

# Variational inference

---

class: middle

## Latent variable model

.center.width-20[![](figures/lec11/latent-model.svg)]

Consider for now a **prescribed latent variable model** that relates a set of observable variables $\mathbf{x} \in \mathcal{X}$ to a set of unobserved variables $\mathbf{z} \in \mathcal{Z}$.

???

The probabilistic model is given and motivated by domain knowledge assumptions.

Examples include:
- Linear discriminant analysis
- Bayesian networks
- Hidden Markov models
- Probabilistic programs

---

class: middle, black-slide

.center[<video controls autoplay loop muted preload="auto" height="480" width="640">
  <source src="./figures/lec11/galton.mp4" type="video/mp4">
</video>]

---

class: middle

The probabilistic model defines a joint probability distribution $p\_\theta(\mathbf{x}, \mathbf{z})$, which decomposes as
$$p\_\theta(\mathbf{x}, \mathbf{z}) = p\_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}).$$
If we interpret $\mathbf{z}$ as causal factors for the high-dimension representations $\mathbf{x}$, then
sampling from $p\_\theta(\mathbf{x}|\mathbf{z})$ can be interpreted as **a stochastic generating process** from $\mathcal{Z}$ to $\mathcal{X}$.

---

<br><br><br>

## How to fit a latent variable model?

$$\begin{aligned}
\theta^{\*} &= \arg \max\_\theta p\_\theta(\mathbf{x}) \\\\
&= \arg \max\_\theta \int p\_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}\\\\
&= \arg \max\_\theta \mathbb{E}\_{p(\mathbf{z})}\left[ p\_\theta(\mathbf{x}|\mathbf{z}) \right] d\mathbf{z}\\\\
&\approx \arg \max\_\theta \frac{1}{N} \sum\_{i=1}^N p\_\theta(\mathbf{x}|\mathbf{z}\_i) 
\end{aligned}$$

--

count: false

.alert[The curse of dimensionality will lead to poor estimates of the expectation.]

---

class: middle

## Variational inference

Let us instead consider a variational approach to fit the model parameters $\theta$.

Using a **variational distribution** $q\_\phi(\mathbf{z})$ over the latent variables $\mathbf{z}$, we have
$$\begin{aligned}
\log p\_\theta(\mathbf{x}) &= \log \mathbb{E}\_{p(\mathbf{z})}\left[ p\_\theta(\mathbf{x}|\mathbf{z}) \right]  \\\\
&= \log \mathbb{E}\_{q\_\phi(\mathbf{z})}\left[ \frac{p\_\theta(\mathbf{x}|\mathbf{z})  p(\mathbf{z})}{q\_\phi(\mathbf{z})} \right] \\\\
&\geq \mathbb{E}\_{q\_\phi(\mathbf{z})}\left[ \log \frac{p\_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q\_\phi(\mathbf{z})}  \right] \quad (\text{ELBO}(\mathbf{x};\theta, \phi)) \\\\
&= \mathbb{E}\_{q\_\phi(\mathbf{z})}\left[ \log p\_\theta(\mathbf{x}|\mathbf{z}) \right] - \text{KL}(q\_\phi(\mathbf{z}) || p(\mathbf{z}))
\end{aligned}$$

---

class: middle

Using the Bayes rule, we can also write
$$\begin{aligned}
\text{ELBO}(\mathbf{x};\theta, \phi) &= \mathbb{E}\_{q\_\phi(\mathbf{z})}\left[ \log \frac{p\_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q\_\phi(\mathbf{z})} \right] \\\\
&= \mathbb{E}\_{q\_\phi(\mathbf{z})}\left[ \log \frac{p\_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q\_\phi(\mathbf{z})} \frac{p\_\theta(\mathbf{x})}{p\_\theta(\mathbf{x})} \right] \\\\
&= \mathbb{E}\_{q\_\phi(\mathbf{z})}\left[ \log \frac{p\_\theta(\mathbf{z}|\mathbf{x})}{q\_\phi(\mathbf{z})} p\_\theta(\mathbf{x}) \right] \\\\
&= \log p\_\theta(\mathbf{x}) - \text{KL}(q\_\phi(\mathbf{z}) || p\_\theta(\mathbf{z}|\mathbf{x})).
\end{aligned}$$

Therefore, $\log p\_\theta(\mathbf{x}) = \text{ELBO}(\mathbf{x};\theta, \phi) + \text{KL}(q\_\phi(\mathbf{z}) || p\_\theta(\mathbf{z}|\mathbf{x}))$.

---

class: middle

.center.width-70[![](figures/lec11/elbo.png)]

Provided the KL gap remains small, the model parameters can now be optimized by maximizing the ELBO,
$$\theta^{\*}, \phi^{\*} = \arg \max\_{\theta,\phi} \text{ELBO}(\mathbf{x};\theta,\phi).$$

---

count: false
class: middle

# Variational auto-encoders

---

class: middle

.center[![](figures/lec12/diagram-vae.svg)]

So far we assumed a prescribed probabilistic model motivated by domain knowledge.
We will now directly learn a stochastic generating process $p\_\theta(\mathbf{x}|\mathbf{z})$ with a neural network.

We will also amortize the inference process by learning a second neural network $q\_\phi(\mathbf{z}|\mathbf{x})$ approximating the posterior, conditionally on the observed data $\mathbf{x}$.

---

class: middle

## Variational auto-encoders

A variational auto-encoder is a deep latent variable model where:
- The prior $p(\mathbf{z})$ is prescribed, and usually chosen to be Gaussian.
- The density $p\_\theta(\mathbf{x}|\mathbf{z})$ is parameterized with a **generative network** $\text{NN}\_\theta$
(or decoder) that takes as input $\mathbf{z}$ and outputs parameters to the data distribution. E.g.,
$$\begin{aligned}
\mu, \sigma^2 &= \text{NN}\_\theta(\mathbf{z}) \\\\
p\_\theta(\mathbf{x}|\mathbf{z}) &= \mathcal{N}(\mathbf{x}; \mu, \sigma^2\mathbf{I})
\end{aligned}$$
- The approximate posterior $q\_\phi(\mathbf{z}|\mathbf{x})$ is parameterized
with an **inference network** $\text{NN}\_\phi$ (or encoder) that takes as input $\mathbf{x}$ and
outputs parameters to the approximate posterior. E.g.,
$$\begin{aligned}
\mu, \sigma^2 &= \text{NN}\_\phi(\mathbf{x}) \\\\
q\_\phi(\mathbf{z}|\mathbf{x}) &= \mathcal{N}(\mathbf{z}; \mu, \sigma^2\mathbf{I})
\end{aligned}$$

---

class: middle

As before, we can use variational inference to jointly optimize the generative and the inference networks parameters $\theta$ and $\phi$:
$$\begin{aligned}
\theta^{\*}, \phi^{\*} &= \arg \max\_{\theta,\phi} \mathbb{E}\_{p(\mathbf{x})} \left[ \text{ELBO}(\mathbf{x};\theta,\phi) \right] \\\\
&= \arg \max\_{\theta,\phi} \mathbb{E}\_{p(\mathbf{x})}\left[ \mathbb{E}\_{q\_\phi(\mathbf{z}|\mathbf{x})}\left[ \log p\_\theta(\mathbf{x}|\mathbf{z})\right] - \text{KL}(q\_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) \right].
\end{aligned}$$

???

- Given some generative network $\theta$, we want to put the mass of the latent variables, by adjusting $\phi$, such that they explain the observed data, while remaining close to the prior.
- Given some inference network $\phi$, we want to put the mass of the observed variables, by adjusting $\theta$, such that
they are well explained by the latent variables.

---

class: middle

## Step-by-step example

Consider as data $\mathbf{d}$ the MNIST digit dataset:

.center.width-100[![](figures/lec11/mnist.png)]

---

class: middle

.italic[Generative network:]
$$\begin{aligned}
\mathbf{z} &\in \mathbb{R}^d \\\\
p(\mathbf{z}) &= \mathcal{N}(\mathbf{z}; \mathbf{0},\mathbf{I})\\\\
p\_\theta(\mathbf{x}|\mathbf{z}) &= \mathcal{N}(\mathbf{x};\mu(\mathbf{z};\theta), \sigma^2(\mathbf{z};\theta)\mathbf{I}) \\\\
\mu(\mathbf{z};\theta) &= \mathbf{W}\_2^T\mathbf{h} + \mathbf{b}\_2 \\\\
\log \sigma^2(\mathbf{z};\theta) &= \mathbf{W}\_3^T\mathbf{h} + \mathbf{b}\_3 \\\\
\mathbf{h} &= \text{ReLU}(\mathbf{W}\_1^T \mathbf{z} + \mathbf{b}\_1)\\\\
\theta &= \\\{ \mathbf{W}\_1, \mathbf{b}\_1, \mathbf{W}\_2, \mathbf{b}\_2, \mathbf{W}\_3, \mathbf{b}\_3 \\\}
\end{aligned}$$

---

class: middle

.italic[Inference network:]
$$\begin{aligned}
q\_\phi(\mathbf{z}|\mathbf{x}) &=  \mathcal{N}(\mathbf{z};\mu(\mathbf{x};\phi), \sigma^2(\mathbf{x};\phi)\mathbf{I}) \\\\
p(\epsilon) &= \mathcal{N}(\epsilon; \mathbf{0}, \mathbf{I}) \\\\
\mathbf{z} &= \mu(\mathbf{x};\phi) + \sigma(\mathbf{x};\phi) \odot \epsilon \\\\
\mu(\mathbf{x};\phi) &= \mathbf{W}\_5^T\mathbf{h} + \mathbf{b}\_5 \\\\
\log \sigma^2(\mathbf{x};\phi) &= \mathbf{W}\_6^T\mathbf{h} + \mathbf{b}\_6 \\\\
\mathbf{h} &= \text{ReLU}(\mathbf{W}\_4^T \mathbf{x} + \mathbf{b}\_4)\\\\
\phi &= \\\{ \mathbf{W}\_4, \mathbf{b}\_4, \mathbf{W}\_5, \mathbf{b}\_5, \mathbf{W}\_6, \mathbf{b}\_6 \\\}
\end{aligned}$$

Note that there is no restriction on the generative and inference network architectures.

---

class: middle

Using the reparameterization trick, the objective can be expressed as:
$$\begin{aligned}
& \mathbb{E}\_{p(\mathbf{x})}\left[ \text{ELBO}(\mathbf{x};\theta,\phi) \right] \\\\
&= \mathbb{E}\_{p(\mathbf{x})}\left[ \mathbb{E}\_{q\_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p\_\theta(\mathbf{x}|\mathbf{z}) \right] - \text{KL}(q\_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) \right] \\\\
&= \mathbb{E}\_{p(\mathbf{x})}\left[ \mathbb{E}\_{p(\epsilon)} \left[  \log p(\mathbf{x}|\mathbf{z}=g(\phi,\mathbf{x},\epsilon);\theta) \right] - \text{KL}(q\_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) \right]
\end{aligned}
$$
where the negative KL divergence can be expressed  analytically as
$$-\text{KL}(q\_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) = \frac{1}{2} \sum\_{j=1}^d \left( 1 + \log(\sigma\_j^2(\mathbf{x};\phi)) - \mu\_j^2(\mathbf{x};\phi) - \sigma\_j^2(\mathbf{x};\phi)\right),$$
which allows to evaluate its derivative without approximation.

---

class: middle, center

.width-100[![](figures/lec11/vae-samples.png)]

(Kingma and Welling, 2013)

---

count: false
class: middle

# Hierarchical variational auto-encoders

---

class: middle

The prior matching term $\mathbb{E}\_{p(\mathbf{x})}\left[ \text{KL}(q\_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) \right]$ limits the expressivity of the model.

Solution: Make $p(\mathbf{z})$ a learnable distribution.

???

Explain the maths on the black board, taking the expectation wrt $p(\mathbf{x})$ of the ELBO and consider the expected KL terms.

---

class: middle, black-slide, center
count: false

.width-80[![](figures/lec12/deeper.jpg)]

---

class: middle

## (Markovian) Hierarchical VAEs

The prior $p(\mathbf{z})$ is itself a VAE, and recursively so for its own hyper-prior.

.center[![](figures/lec12/diagram-hvae.svg)]

---

class: middle

Similarly to VAEs, training is done by maximizing the ELBO, using a variational distribution $q\_\phi(\mathbf{z}\_{1:T} | \mathbf{x})$ over all levels of latent variables.

$$\begin{aligned}
\log p\_\theta(\mathbf{x}) &\geq \mathbb{E}\_{q\_\phi(\mathbf{z}\_{1:T} | \mathbf{x})}\left[ \log \frac{p(\mathbf{x},\mathbf{z}\_{1:T})}{q\_\phi(\mathbf{z}\_{1:T}|\mathbf{x})} \right] \\\\
\end{aligned}$$

---

class: middle
count: false

# Variational diffusion models

---

class: middle

.center.width-100[![](figures/lec12/sohl-dickstein2015.png)]

---

class: middle

Variational diffusion models are Markovian HVAEs with the following constraints:
- The latent dimension is the same as the data dimension.
- The encoder is fixed to linear Gaussian transitions $q(\mathbf{x}\_t | \mathbf{x}\_{t-1})$.
- The hyper-parameters are set such that $q(\mathbf{x}_T | \mathbf{x}_0)$ is a standard Gaussian. 

<br>

.center.width-100[![](figures/lec12/vdm.png)]

.footnote[Credits: [Kreis et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

---

class: middle

## Forward diffusion process

.center.width-100[![](figures/lec12/vdm-forward.png)]

With $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we have
$$\begin{aligned}
\mathbf{x}\_t &= \sqrt{ {\alpha}\_t} \mathbf{x}\_{t-1} + \sqrt{1-{\alpha}\_t} \epsilon \\\\
q(\mathbf{x}\_t | \mathbf{x}\_{t-1}) &= \mathcal{N}(\mathbf{x}\_t ; \sqrt{\alpha\_t} \mathbf{x}\_{t-1}, (1-\alpha\_t)\mathbf{I}) \\\\
q(\mathbf{x}\_{1:T} | \mathbf{x}\_{0}) &=  \prod\_{t=1}^T q(\mathbf{x}\_t | \mathbf{x}\_{t-1}) 
\end{aligned}$$

.footnote[Credits: [Kreis et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

???

Start drawing the full probabilistic graphical model as the forward and reverse processes are presented.

---

class: middle

.center.width-100[![](figures/lec12/vdm-forward2.png)]

.footnote[Credits: [Simon J.D. Prince](https://udlbook.github.io/udlbook/), 2023.]

---

class: middle

## Diffusion kernel

.center.width-100[![](figures/lec12/vdm-kernel.png)]

With $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$ and $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we have

$$\begin{aligned}
\mathbf{x}\_t &= \sqrt{\bar{\alpha}\_t} \mathbf{x}\_{0} + \sqrt{1-\bar{\alpha}\_t} \epsilon \\\\
q(\mathbf{x}\_t | \mathbf{x}\_{0}) &= \mathcal{N}(\mathbf{x}\_t ; \sqrt{\bar{\alpha}\_t} \mathbf{x}\_{0}, (1-\bar{\alpha}\_t)\mathbf{I})
\end{aligned}$$

.footnote[Credits: [Kreis et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

---

class: middle

.center.width-100[![](figures/lec12/diffusion-kernel-1.png)]

.footnote[Credits: [Simon J.D. Prince](https://udlbook.github.io/udlbook/), 2023.]

---

class: middle

## Reverse denoising process

.center.width-100[![](figures/lec12/vdm-reverse.png)]

$$\begin{aligned}
p(\mathbf{x}\_{0:T}) &= p(\mathbf{x}\_T) \prod\_{t=1}^T p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t)\\\\
p(\mathbf{x}\_T) &= \mathcal{N}(\mathbf{x}\_T; \mathbf{0}, I) \\\\
p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t) &= \mathcal{N}(\mathbf{x}\_{t-1}; \mu\_\theta(\mathbf{x}\_t, t), \sigma^2\_\theta(\mathbf{x}\_t, t)\mathbf{I}) \\\\
\mathbf{x}\_{t-1} &= \mu\_\theta(\mathbf{x}\_t, t) + \sigma\_\theta(\mathbf{x}\_t, t) \mathbf{z} 
\end{aligned}$$
with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

.footnote[Credits: [Kreis et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

---

class: middle

## Training

For learning the parameters $\theta$ of the reverse process, we can form a variational lower bound on the log-likelihood of the data as 

$$\mathbb{E}\_{q(\mathbf{x}\_0)}\left[ \log p\_\theta(\mathbf{x}\_0) \right] \geq \mathbb{E}\_{q(\mathbf{x}\_0)q(\mathbf{x}\_{1:T}|\mathbf{x}\_0)}\left[ \log \frac{p\_\theta(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T} | \mathbf{x}\_0)} \right] := L$$

???

Derive on the board.

---

class: middle

This objective can be rewritten as
$$\begin{aligned}
L &= \mathbb{E}\_{q(\mathbf{x}\_0)q(\mathbf{x}\_{1:T}|\mathbf{x}\_0)}\left[ \log \frac{p\_\theta(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T} | \mathbf{x}\_0)} \right] \\\\
&= \mathbb{E}\_{q(\mathbf{x}\_0)} \left[L\_0 - \sum\_{t>1} L\_{t-1} - L\_T\right]
\end{aligned}$$
where
- $L\_0 = \mathbb{E}\_{q(\mathbf{x}\_1 | \mathbf{x}\_0)}[\log p\_\theta(\mathbf{x}\_0 | \mathbf{x}\_1)]$ can be interpreted as a reconstruction term. It can be approximated and optimized using a Monte Carlo estimate.
- $L\_{t-1} = \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)}\text{KL}(q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0) || p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t) )$ is a denoising matching term. The transition $q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0)$ provides a learning signal for the reverse process, since it defines how to denoise the noisified input $\mathbf{x}\_t$ with access to the original input $\mathbf{x}\_0$.
- $L\_T = \text{KL}(q(\mathbf{x}\_T | \mathbf{x}\_0) || p\_\theta(\mathbf{x}\_T))$ represents how close the distribution of the final noisified input is to the standard Gaussian. It has no trainable parameters.

---

class: middle

.center[![](figures/lec12/tractable-posterior.svg)]

The distribution $q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0)$ is the tractable posterior distribution
$$\begin{aligned}
q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0) &= \frac{q(\mathbf{x}\_t | \mathbf{x}\_{t-1}, \mathbf{x}\_0) q(\mathbf{x}\_{t-1} | \mathbf{x}\_0)}{q(\mathbf{x}\_t | \mathbf{x}\_0)} \\\\
&= \mathcal{N}(\mathbf{x}\_{t-1}; \mu\_q(\mathbf{x}\_t, \mathbf{x}\_0, t), \sigma^2\_t I)
\end{aligned}$$
where
$$\begin{aligned}
\mu\_q(\mathbf{x}\_t, \mathbf{x}\_0, t) &= \frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\mathbf{x}\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}(1-\alpha\_t)}{1-\bar{\alpha}\_t}\mathbf{x}\_0 \\\\
\sigma^2\_t &= \frac{(1-\alpha\_t)(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}
\end{aligned}$$

???

Take the time to do the derivation on the board.

---

class: middle

## Interpretation 1: Denoising

To minimize the expected KL divergence $L\_{t-1}$, we need to match the reverse process $p\_\theta(\mathbf{x}\_{t-1}|\mathbf{x}\_t)$ to the tractable posterior. Since both are Gaussian, we can match their means and variances.

By construction, the variance of the reverse process can be set to the known variance $\sigma^2\_t$ of the tractable posterior.

For the mean, we reuse the analytical form of $\mu\_q(\mathbf{x}\_t, \mathbf{x}\_0, t)$ and parameterize the mean of the reverse process using a .bold[denoising network] as
$$\mu\_\theta(\mathbf{x}\_t, t) = \frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\mathbf{x}\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}(1-\alpha\_t)}{1-\bar{\alpha}\_t}\hat{\mathbf{x}}\_\theta(\mathbf{x}\_t, t).$$

???

Derive on the board.

---

class: middle

Under this parameterization, the minimization of expected KL divergence $L\_{t-1}$ can be rewritten as
$$\begin{aligned}
&\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)}\text{KL}(q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0) || p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t) )\\\\
=&\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)} \frac{1}{2\sigma^2\_t} || \mu\_\theta(\mathbf{x}\_t, t) - \mu\_q(\mathbf{x}\_t, \mathbf{x}\_0, t) ||\_2^2 \\\\
=&\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)} \frac{1}{2\sigma^2\_t} \frac{\bar{\alpha}\_{t-1}(1-\alpha\_t)^2}{(1-\bar{\alpha}\_t)^2} || \hat{\mathbf{x}}\_\theta(\mathbf{x}\_t, t) - \mathbf{x}\_0 ||\_2^2
\end{aligned}$$

.success[Optimizing a VDM amounts to learning a neural network that predicts the original ground truth $\mathbf{x}\_0$ from a noisy input $\mathbf{x}\_t$.]

---

class: middle

Finally, minimizing the summation of the $L\_{t-1}$ terms across all noise levels $t$ can be approximated by minimizing the expectation over all timesteps as
$$\arg \min\_\theta \mathbb{E}\_{t \sim U\\{2,T\\}} \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)}\text{KL}(q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0) || p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t) ).$$

---

class: middle

## Interpretation 2: Noise prediction

A second interpretation of VDMs can be obtained using the reparameterization trick. 
Using $$\mathbf{x}\_0 = \frac{\mathbf{x}\_t - \sqrt{1-\bar{\alpha}\_t} \epsilon}{\sqrt{\bar{\alpha}\_t}},$$
we can rewrite the mean of the tractable posterior as
$$\begin{aligned}
\mu\_q(\mathbf{x}\_t, \mathbf{x}\_0, t) &= \frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\mathbf{x}\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}(1-\alpha\_t)}{1-\bar{\alpha}\_t}\mathbf{x}\_0 \\\\
&= \frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\mathbf{x}\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}(1-\alpha\_t)}{1-\bar{\alpha}\_t}\frac{\mathbf{x}\_t - \sqrt{1-\bar{\alpha}\_t} \epsilon}{\sqrt{\bar{\alpha}\_t}} \\\\
&= ... \\\\
&= \frac{1}{\sqrt{\alpha}\_t} \mathbf{x}\_t - \frac{1-\alpha\_t}{\sqrt{(1-\bar{\alpha}\_t)\alpha\_t}}\epsilon
\end{aligned}$$

???

Derive on the board.

---

class: middle

Accordingly, the mean of the reverse process can be parameterized with a .bold[noise-prediction network] as

$$\mu\_\theta(\mathbf{x}\_t, t) = \frac{1}{\sqrt{\alpha}\_t} \mathbf{x}\_t - \frac{1-\alpha\_t}{\sqrt{(1-\bar{\alpha}\_t)\alpha\_t}}{\epsilon}\_\theta(\mathbf{x}\_t, t).$$

Under this parameterization, the minimization of the expected KL divergence $L\_{t-1}$ can be rewritten as
$$\begin{aligned}
&\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)}\text{KL}(q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0) || p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t) )\\\\
=&\arg \min\_\theta \mathbb{E}\_{\mathcal{N}(\epsilon;\mathbf{0}, I)} \frac{1}{2\sigma^2\_t} \frac{(1-\alpha\_t)^2}{(1-\bar{\alpha}\_t) \alpha\_t} || {\epsilon}\_\theta(\underbrace{\sqrt{\bar{\alpha}\_t} \mathbf{x}\_{0} + \sqrt{1-\bar{\alpha}\_t} \epsilon}\_{\mathbf{x}\_t}, t) - \epsilon ||_2^2
\end{aligned}$$

.success[Optimizing a VDM amounts to learning a neural network that predicts the noise $\epsilon$ that was added to the original ground truth $\mathbf{x}\_0$ to obtain the noisy $\mathbf{x}\_t$.]

---

class: middle

## Algorithms

.center.width-100[![](figures/lec12/algorithms.png)]

???

Note that in practice, the coefficient before the norm in the loss function is often omitted. Setting it to 1 is found to increase the sample quality.

---

class: middle

## Network architectures

Diffusion models often use U-Net architectures with ResNet blocks and self-attention layers to represent $\epsilon\_\theta(\mathbf{x}\_t, t)$.

<br>

.center.width-100[![](figures/lec12/architecture.png)]

.footnote[Credits: [Kreis et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

---

class: middle

# Score-based generative models

---

class: middle

## The score function

The score function $\nabla\_{\mathbf{x}\_0} \log q(\mathbf{x}\_0)$ is a vector field that points in the direction of the highest density of the data distribution $q(\mathbf{x}\_0)$.

It can be used to find modes of the data distribution or to generate samples by Langevin dynamics.

.center.width-40[![](figures/lec12/langevin.gif)]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle

## Interpretation 3: Denoising score matching

A third interpretation of VDMs can be obtained by reparameterizing $\mathbf{x}\_0$ using Tweedie's formula, as
$$\mathbf{x}\_0 = \frac{\mathbf{x}\_t + (1-\bar{\alpha}\_t) \nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t | \mathbf{x}\_0) }{\sqrt{\bar{\alpha}\_t}},$$
which we can plug into the the mean of the tractable posterior to obtain
$$\begin{aligned}
\mu\_q(\mathbf{x}\_t, \mathbf{x}\_0, t) &= \frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\mathbf{x}\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}(1-\alpha\_t)}{1-\bar{\alpha}\_t}\mathbf{x}\_0 \\\\
&= ... \\\\
&= \frac{1}{\sqrt{\alpha}\_t} \mathbf{x}\_t + \frac{1-\alpha\_t}{\sqrt{\alpha\_t}} \nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t | \mathbf{x}\_0).
\end{aligned}$$

???

Derive on the board.

---

class: middle

The mean of the reverse process can be parameterized with a .bold[score network] as
$$\mu\_\theta(\mathbf{x}\_t, t) = \frac{1}{\sqrt{\alpha}\_t} \mathbf{x}\_t + \frac{1-\alpha\_t}{\sqrt{\alpha\_t}} s\_\theta(\mathbf{x}\_t, t).$$

Under this parameterization, the minimization of the expected KL divergence $L\_{t-1}$ can be rewritten as
$$\begin{aligned}
&\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)}\text{KL}(q(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{x}\_0) || p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t) )\\\\
=&\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)} \frac{1}{2\sigma^2\_t} \frac{(1-\alpha\_t)^2}{\alpha\_t} || s\_\theta(\mathbf{x}\_t, t) - \nabla\_{\mathbf{x}\_t}  \log q(\mathbf{x}\_t | \mathbf{x}\_0) ||_2^2
\end{aligned}$$

.success[Optimizing a score-based model amounts to learning a neural network that predicts the score $\nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t | \mathbf{x}\_0)$  of the tractable posterior.]

---

class: middle

Since $s\_\theta(\mathbf{x}\_t, t)$ is learned in expectation over the data distribution $q(\mathbf{x}\_0)$, the score network will eventually approximate the score of the marginal distribution $q(\mathbf{x}\_t$), for each noise level $t$, that is
$$s\_\theta(\mathbf{x}\_t, t) \approx \nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t).$$

---

class: middle

## Ancestral sampling

Sampling from the score-based diffusion model is done by starting from $\mathbf{x}\_T \sim p(\mathbf{x}\_T)=\mathcal{N}(\mathbf{0}, \mathbf{I})$ and then following the estimated reverse Markov chain, as
$$\mathbf{x}\_{t-1} = \frac{1}{\sqrt{\alpha}\_t} \mathbf{x}\_t + \frac{1-\alpha\_t}{\sqrt{\alpha\_t}} s\_\theta(\mathbf{x}\_t, t) + \sigma\_t \mathbf{z}\_t,$$
where $\mathbf{z}\_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, for $t=T, ..., 1$. 

---

class: middle

## Conditional sampling

To turn a diffusion model $p\_\theta(\mathbf{x}\_{0:T})$ into a conditional model, we can add conditioning information $y$ at each step of the reverse process, as
$$p\_\theta(\mathbf{x}\_{0:T} | y) = p(\mathbf{x}\_T) \prod\_{t=1}^T p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}\_t, y).$$

---

class: middle

With a score-based model however, we can use the Bayes rule and notice that
$$\nabla\_{\mathbf{x}\_t} \log p(\mathbf{x}\_t | y) = \nabla\_{\mathbf{x}\_t} \log p(\mathbf{x}\_t) + \nabla\_{\mathbf{x}\_t} \log p(y | \mathbf{x}\_t),$$
where we leverage the fact that the gradient of $\log p(y)$ with respect to $\mathbf{x}\_t$ is zero.

In other words, controllable generation can be achieved by adding a conditioning signal during sampling, without having to retrain the model. E.g., train an extra classifier $p(y | \mathbf{x}\_t)$ and use it to control the sampling process by adding its gradient to the score.

---

class: middle

## Continuous-time diffusion models

.center.width-100[![](figures/lec12/vdm-forward.png)]

With $\beta\_t = 1 - \alpha\_t$, we can rewrite the forward process as
$$\begin{aligned}
\mathbf{x}\_t &= \sqrt{ {\alpha}\_t} \mathbf{x}\_{t-1} + \sqrt{1-{\alpha}\_t} \mathcal{N}(\mathbf{0}, \mathbf{I}) \\\\
&= \sqrt{1 - {\beta}\_t} \mathbf{x}\_{t-1} + \sqrt{ {\beta}\_t} \mathcal{N}(\mathbf{0}, \mathbf{I}) \\\\
&= \sqrt{1 - {\beta}(t)\Delta\_t} \mathbf{x}\_{t-1} + \sqrt{ {\beta}(t)\Delta\_t} \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{aligned}$$

.footnote[Credits: [Kreis et al](https://cvpr2022-tutorial-diffusion-models.github.io/), 2022.]

---

class: middle

In the limit of many small steps, i.e. as $\Delta\_t \rightarrow 0$, we can further rewrite the forward process as
$$\begin{aligned}
\mathbf{x}\_t &= \sqrt{1 - {\beta}(t)\Delta\_t} \mathbf{x}\_{t-1} + \sqrt{ {\beta}(t)\Delta\_t} \mathcal{N}(\mathbf{0}, \mathbf{I}) \\\\
&\approx \mathbf{x}\_{t-1} - \frac{\beta(t)\Delta\_t}{2} \mathbf{x}\_{t-1} + \sqrt{ {\beta}(t)\Delta\_t} \mathcal{N}(\mathbf{0}, \mathbf{I}) 
\end{aligned}.$$

This last update rule corresponds to the Euler-Maruyama discretization of the stochastic differential equation (SDE)
$$\text{d}\mathbf{x}\_t = -\frac{1}{2}\beta(t)\mathbf{x}\_t \text{d}t + \sqrt{\beta(t)} \text{d}\mathbf{w}\_t$$
describing the diffusion in the infinitesimal limit.

---

class: middle

.center.width-80[![](figures/lec12/perturb_vp.gif)]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle

The reverse process satisfies a reverse-time SDE that can be derived analytically from the forward-time SDE and the score of the marginal distribution $q(\mathbf{x}\_t)$, as
$$\text{d}\mathbf{x}\_t = \left[ -\frac{1}{2}\beta(t)\mathbf{x}\_t - \beta(t)\nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t) \right] \text{d}t + \sqrt{\beta(t)} \text{d}\mathbf{w}\_t.$$

---

class: middle

.center.width-80[![](figures/lec12/denoise_vp.gif)]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle

The score $\nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t)$ of the marginal diffused density $q(\mathbf{x}\_t)$ is not tractable, but can be estimated using denoising score matching (DSM) by solving
$$\arg \min\_\theta \mathbb{E}\_{q(\mathbf{x}\_0)} \mathbb{E}\_{t\sim U[0,T]} \mathbb{E}\_{q(\mathbf{x}\_t | \mathbf{x}\_0)} || s\_\theta(\mathbf{x}\_t, t) - \nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t | \mathbf{x}\_0) ||\_2^2,$$
which will result in $s\_\theta(\mathbf{x}\_t, t) \approx \nabla\_{\mathbf{x}\_t} \log q(\mathbf{x}\_t)$ because of the outer expectation over $q(\mathbf{x}\_0)$.

.success[This is just the .bold[same objective] as for VDMs! (See Interpretation 3)]

---

class: middle

## Latent-space diffusion models

Directly modeling the data distribution can be make the denoising process difficult to learn. A more effective approach is to combine VAEs with a diffusion prior.
- The distribution of latent embeddings is simpler to model.
- Diffusion on non-image data is possible with tailored autoencoders.

<br>

.center.width-100[![](figures/lec12/lsgm.png)]

.footnote[Credits: [Vahdat et al](https://nvlabs.github.io/LSGM/), 2021.]

---

class: black-slide, middle
count: false

.center[

<video autoplay muted loop width="500" height="300">
     <source src="./figures/lec12/teddy_bear_guitar.mp4" type="video/mp4">
</video>

The end.

]

.footnote[Credits: [Blattmann et al](https://research.nvidia.com/labs/toronto-ai/VideoLDM/), 2023. Prompt: "A teddy bear is playing the electric guitar, high definition, 4k."]



