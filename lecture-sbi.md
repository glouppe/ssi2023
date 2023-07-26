class: middle, center, title-slide

# An introduction to<br>simulation-based inference

51st SLAC Summer Institute

August 16, 2023

<br>

Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

---

class: middle, center

.center.width-80[![](./figures/sbi/pitcher.png)]

---

class: middle

$$
v_x = v \cos(\alpha),~~ v_y = v \sin(\alpha),
$$

$$
\frac{dx}{dt} = v_x,~~\frac{dy}{dt} = v_y, \frac{dv_y}{dt} = -G.
$$

---

class: middle

```python
def simulate(v, alpha, dt=0.001):
    v_x = v * np.cos(alpha)  # x velocity m/s
    v_y = v * np.sin(alpha)  # y velocity m/s
    y = 1.1 + 0.3 * random.normal()
    x = 0.0

    while y > 0: # simulate until ball hits floor
        v_y += dt * -G  # acceleration due to gravity
        x += dt * v_x
        y += dt * v_y

    return x + 0.25 * random.normal()
```

---

class: middle, center

.center.width-100[![](./figures/sbi/likelihood-model.png)]

---

class: middle, center

What parameter values $\theta$ are the most plausible?

---

class: middle

.center.width-100[![](./figures/sbi/prior_posterior.png)]

---

# Outline

- Part 1: Simulation-based inference
- Part 2: Algorithms
    - Neural ratio estimation
    - Neural posterior estimation 
    - Neural score estimation
- Part 3: Diagnostics

???

Check https://github.com/smsharma/sbi-lecture-mit/blob/main/tutorial.ipynb

---

class: middle
count: false

# Simulation-based inference

---

class: middle

## Scientific simulators

.center.width-100[![](./figures/sbi/simulators.png)]

---

class: middle

.center.width-70[![](./figures/sbi/unconditioned-program.png)]

.center[$$\theta, z, x \sim p(\theta, z, x)$$]

---

class: middle, center

.center.width-70[![](./figures/sbi/conditioned-program.png)]

.center[$$\theta, z \sim p(\theta, z | x)$$]

???

Take the bean machine example.

---

class: middle

.width-100[![](figures/process1.png)]

???

generation: pencil and paper calculable from first principles

---

count: false
class: middle

.width-100[![](figures/process2.png)]

???

parton shower + hadronization: controlled approximation of first principles + phenomenological model

---

count: false
class: middle

.width-100[![](figures/process3.png)]

???

detector simulation: interaction with the materials and digitization

---

count: false
class: middle

.width-100[![](figures/process4.png)]

???

reconstruction simulation

---

class: middle

$$p(x|\theta) = \underbrace{\iiint}\_{\text{yikes!}} p(z\_p|\theta) p(z\_s|z\_p) p(z\_d|z\_s) p(x|z\_d) dz\_p dz\_s dz\_d$$

???

That's bad!

---

class: middle

## Bayesian inference 

Start with
- a simulator that can generate $N$ samples $x\_i \sim p(x\_i|\theta\_i)$,
- a prior model $p(\theta)$,
- observed data $x\_\text{obs} \sim p(x\_\text{obs} | \theta\_\text{true})$.

Then, estimate the posterior $$p(\theta|x\_\text{obs}) = \frac{p(x\_\text{obs} | \theta)p(\theta)}{p(x\_\text{obs})}.$$

---

class: middle

.center.width-100[![](./figures/sbi/sbi.png)]

---

class: middle
count: false

# Algorithms

---


class: middle

.avatars[![](figures/faces/kyle.png)![](figures/faces/johann.png)]

.center.width-100[![](./figures/sbi/frontiers-sbi0.png)]

.footnote[Credits: [Cranmer, Brehmer and Louppe](https://doi.org/10.1073/pnas.1912789117), 2020.]

---

class: middle

## Approximate Bayesian Computation (ABC)

.center.width-100[![](./figures/sbi/abc.png)]

.italic[Issues:]
- How to choose $x'$? $\epsilon$? $||\cdot||$?
- No tractable posterior.
- Need to run new simulations for new data or new prior.

.footnote[Credits: Johann Brehmer.]

---

class: middle

.avatars[![](figures/faces/kyle.png)![](figures/faces/johann.png)]

.center.width-100[![](./figures/sbi/frontiers-sbi2.png)]

.footnote[Credits: [Cranmer, Brehmer and Louppe](https://doi.org/10.1073/pnas.1912789117), 2020.]

---

class: middle
count: false

.avatars[![](figures/faces/kyle.png)![](figures/faces/johann.png)]

.center.width-100[![](./figures/sbi/frontiers-sbi.png)]

.footnote[Credits: [Cranmer, Brehmer and Louppe](https://doi.org/10.1073/pnas.1912789117), 2020.]

---

# Neural ratio estimation 

.avatars[![](figures/faces/kyle.png)![](figures/faces/joeri.png)![](figures/faces/johann.png)]

<br>
The likelihood-to-evidence $r(x|\theta) = \frac{p(x|\theta)}{p(x)} = \frac{p(x, \theta)}{p(x)p(\theta)}$ ratio can be learned, even if neither the likelihood nor the evidence can be evaluated:
<br><br>
.grid[
.kol-1-4.center[

<br>

$x,\theta \sim p(x,\theta)$

<br><br><br><br>

$x,\theta \sim p(x)p(\theta)$

]
.kol-5-8[<br>.center.width-70[![](./figures/sbi/classification-2.png)]]
.kol-1-8[<br><br><br><br>

$\hat{r}(x|\theta)$]
]

.footnote[Credits: [Cranmer et al](https://arxiv.org/pdf/1506.02169.pdf), 2015; [Hermans et al](http://proceedings.mlr.press/v119/hermans20a/hermans20a.pdf), 2020.]

---

class: middle

.avatars[![](figures/faces/kyle.png)![](figures/faces/joeri.png)![](figures/faces/johann.png)]

The solution $d$ found after training  approximates the optimal classifier
$$d(x, \theta) \approx d^\*(x, \theta) = \frac{p(x, \theta)}{p(x, \theta)+p(x)p(\theta)}.$$
Therefore, $$r(x|\theta) = \frac{p(x|\theta)}{p(x)} = \frac{p(x, \theta)}{p(x)p(\theta)} \approx \frac{d(x, \theta)}{1-d(x, \theta)} = \hat{r}(x|\theta).$$

???

Derive on blackboard.

---

class: middle

.avatars[![](figures/faces/kyle.png)![](figures/faces/joeri.png)![](figures/faces/johann.png)]

.center.width-100[![](./figures/sbi/carl.png)]

$$p(\theta|x) \approx \hat{r}(x|\theta) p(\theta) $$

---

background-image: url(./figures/sbi/stellar.jpeg)
background-position: left
class: black-slide

.avatars[![](figures/faces/joeri.png)![](figures/faces/nil.jpg)![](figures/faces/christoph.jpg)![](figures/faces/gf.jpg)]

.smaller-x[ ]
## Constraining dark matter with stellar streams 

<br><br><br><br><br><br>
.pull-right[
  
<iframe width="360" height="270" src="https://www.youtube.com/embed/uQVv_Sfxx5E?&loop=1&start=0" frameborder="0" volume="0" allowfullscreen></iframe>

]

.footnote[Image credits: C. Bickel/Science; [D. Erkal](https://t.co/U6KPgLBdpz?amp=1).].]

---

class: middle

.avatars[![](figures/faces/joeri.png)![](figures/faces/nil.jpg)![](figures/faces/christoph.jpg)![](figures/faces/gf.jpg)]

.center.width-90[![](./figures/sbi/dm1.png)]

.center[
.width-35[![](./figures/sbi/posterior-gd1-1d.png)]
.width-35[![](./figures/sbi/posterior-gd1-2d.png)]

]

.footnote[Credits: [Hermans et al](https://arxiv.org/pdf/2011.14923), 2021.]

---

class: black-slide, center, middle

.width-100[![](./figures/sbi/cdm-wdm.jpg)]

Preliminary results for GD-1 suggest a .bold[preference for CDM over WDM].

---

# Neural Posterior Estimation 

<br>

.center.width-90[![](./figures/sbi/fig1.png)]

$$\begin{aligned}
&\min\_{q\_\phi} \mathbb{E}\_{p(x)}\left[ \text{KL}( p(\theta|x) || q\_\phi(\theta|x) ) \right]
\end{aligned}$$

???

Derive on blackboard

$$
\begin{aligned}
&\min\_{q\_\phi} \mathbb{E}\_{p(x)}\left[ \text{KL}( p(\theta|x) || q\_\phi(\theta|x) ) \right] \\\\
=& \min\_{q\_\phi} \mathbb{E}\_{p(x)} \mathbb{E}\_{p(\theta|x)} \left[\log \frac{p(\theta|x)}{q\_\phi(\theta|x)} \right] \\\\
=& \max\_{q\_\phi} \mathbb{E}\_{p(x, \theta)} \left[ \log q\_\phi(\theta|x) \right]
\end{aligned}$$

---

class: middle

## Normalizing flows

A normalizing flow is a sequence of invertible transformations $f\_k$ that map a simple distribution $p\_0$ to a more complex distribution $p\_K$:

.center.width-100[![](./figures/sbi/normalizing-flow.png)]

By the change of variables formula, the log-likelihood of a sample $x$ is given by
$$\log p\_K(x) = \log p\_0(z) - \sum\_{k=1}^K \log \left| \det \frac{\partial f\_k}{\partial z\_{k-1}} \right|.$$

---

class: middle, black-slide

## Exoplanet atmosphere characterization 

.center.width-95[![](./figures/sbi/exoplanet-probe.jpg)]

.footnote[Credits: [NSA/JPL-Caltech](https://www.nasa.gov/topics/universe/features/exoplanet20100203-b.html), 2010.]

---

class: middle

.avatars[![](figures/faces/malavika.jpg)![](figures/faces/francois.jpg)![](figures/faces/absil.jpg)]

.center[
.width-25[![](./figures/sbi/exoplanet-residuals.png)]
.width-70[![](./figures/sbi/exoplanet-corner.png)]
]

.footnote[Credits: [Vasist et al](https://doi.org/10.1051/0004-6361/202245263), 2023.]

---

class: middle
count: false

# Diagnostics

---

class: middle

.grid[
.kol-1-2[
<br>
$$\hat{p}(\theta|x) = \text{sbi}(p(x | \theta), p(\theta), x)$$

We must make sure our approximate simulation-based inference algorithms can (at least) actually realize faithful inferences on the (expected) observations.

]
.kol-1-2[.center.width-80[![](figures/exoplanet-corner.png)]

.center.italic[How do we know this is good enough?]]
]

---

class: middle

.avatars[![](figures/faces/kyle.png)![](figures/faces/johann.png)![](figures/faces/siddarth.png)![](figures/faces/joeri.png)]

## Mode convergence

The maximum a posteriori estimate converges towards the nominal value $\theta^\*$ for an increasing number of independent and identically distributed observables $x\_i \sim p(x|\theta^\*)$:
$$\begin{aligned}
&\lim\_{N \to \infty} \arg\max\_\theta p(\theta | \\{ x\_i \\}\_{i=1}^N) \\\\
=& \lim\_{N \to \infty} \arg\max\_\theta p(\theta) \prod\_{x\_i} r(x\_i | \theta) = \theta^\*
\end{aligned}$$

.center.width-100[![](figures/dm-posterior.gif)]

.footnote[Credits: [Brehmer et al](https://iopscience.iop.org/article/10.3847/1538-4357/ab4c41/meta), 2019.]

---

class: middle

.avatars[![](figures/faces/joeri.png)![](figures/faces/arnaud.jpg)![](figures/faces/francois.jpg)![](figures/faces/antoine.png)]

.grid[
.kol-2-3[

<br>

## Coverage diagnostic

- For $x,\theta \sim p(x,\theta)$, compute the $1-\alpha$ credible interval based on $\hat{p}(\theta|x)$.
- If the fraction of samples for which $\theta$ is contained within the interval is larger than the nominal coverage probability $1-\alpha$, then the approximate posterior $\hat{p}(\theta|x)$ has coverage.]
.kol-1-3[
.center.width-100[![](./figures/sbi/posterior-large-small.png)]
.center.width-95[![](./figures/sbi/coverage.png)]
]
]

.footnote[Credits: [Hermans et al](https://arxiv.org/abs/2110.06581), 2021; [Siddharth Mishra-Sharma](https://arxiv.org/abs/2110.01620), 2021.]

---

class: middle

.avatars[![](figures/faces/joeri.png)![](figures/faces/arnaud.jpg)![](figures/faces/francois.jpg)![](figures/faces/antoine.png)]

<br>

.center.width-90[![](figures/coverage-crisis.png)]

.footnote[Credits: [Hermans et al](https://arxiv.org/abs/2110.06581), 2021.]

---

class: middle, center

What if diagnostics fail?

???

- Use more data
- Use better NN architectures
- Use an ensemble

---

# Balanced NRE

.avatars[![](figures/faces/joeri.png)![](figures/faces/arnaud.jpg)![](figures/faces/francois.jpg)![](figures/faces/antoine.png)]

Enforce neural ratio estimation to be .bold[conservative] by using binary classifiers $\hat{d}$ that are balanced, i.e. such that
$$
\mathbb{E}\_{p(\theta,x)}\left[\hat{d}(\theta,x)\right] = \mathbb{E}\_{p(\theta)p(x)}\left[1 - \hat{d}(\theta,x)\right].
$$

<br>

.center.width-100[![](figures/mainbnre1.png)]

.footnote[Credits: [Delaunoy et al](https://arxiv.org/abs/2208.13624), 2022.]

---

class: middle

.avatars[![](figures/faces/joeri.png)![](figures/faces/arnaud.jpg)![](figures/faces/francois.jpg)![](figures/faces/antoine.png)]

.center.width-105[![](figures/bnre-effect.png)]

.footnote[Credits: [Delaunoy et al](https://arxiv.org/abs/2208.13624), 2022.]

---

# Summary

.success[Advances in deep learning have enabled new approaches to statistical inference.]

.success[This is major evolution in the statistical capabilities for science, as it enables the analysis of complex models and data without simplifying assumptions.]

.alert[Inference remains approximate and requires careful validation.]

.alert[Obstacles remain to be overcome, such as the curse of dimensionality and the need for large amounts of data.]

---

class: end-slide, center, middle
count: false

The end.