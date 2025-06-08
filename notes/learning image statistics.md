model: 
$$
\begin{align}
\mathbf{x}  & \in \mathbb{R}^{M} \\
\mathbf{y} &  \in \mathbb{R}^N \\
\theta  & = \{ \mathbf{W} , \mathbf{b}, \mathbf{c} \}
\end{align}
$$
$$
P(\mathbf{x}, \mathbf{y} \mid \boldsymbol{\theta}) = P(\mathbf{y})P(\mathbf{x}\mid\mathbf{y}, \mathbf{W},\mathbf{b}, \boldsymbol{\sigma}^{2})
$$
$$
\mathbf{x} = \mathbf{b} + \mathbf{W} \mathbf{y} + \boldsymbol{\sigma}\boldsymbol{\epsilon}
$$
note: 
$$
P(\mathbf{y})=\prod_{i} P(y_{i}), P(\mathbf{x}\mid\mathbf{y}, \boldsymbol{\theta}) = \prod_{i}P(x_{i} \mid \mathbf{y}, \boldsymbol{\theta})
$$


Sparse distribution to define the $P(y_{i})$ (kurtosis higher than Gaussian)
such as the Laplace distribution (exponential instead of squared exponential)

- for each observation, $\mathbf{x}$ is in the pixel space, want to account for it by a small set of latent variables

mean field representation (approximation)
$$
\tilde{P}(\mathbf{y} \mid \mathbf{x}; r(t)) = \prod_{i} \tilde{p}(y_{i}\mid\mathbf{x};r_{i}(t))
$$
- the neurons will serve to parametrize the distribution over the y. mean feild: neurons are factorized in y.
- also: use a dirac delta

$$
\tilde{(p)}(y_{i} \mid \mathbf{x}; r_{i}(t)) = \delta(y_i - r_i)
$$


minimize KL when the delta is centered at the maximum of the distribution. 
- MAP inference by neural dynamics: 