	# Proposition 1: Fixed point equations around the ELBO

Denote ELBO as $\mathcal{F}$: 
$$
\begin{align}
q_{\mathcal{F} }^{t}(\mathbf{z}|\mathbf{x})  & \propto \pi(\mathbf{z})e^{ -\lVert \mathbf{x} - g^{t - 1}_{\mathcal{F}(\mathbf{z})} \rVert ^{2}/(2\sigma^{2}) } \\
g^{t}_{\mathcal{F}}(\mathbf{z})  & = \sum_{i} w_{i}^{t} (\mathbf{z})\mathbf{x}_{i}  & w_{i} = \frac{q^{t}_{\mathcal{F}}(\mathbf{z}|\mathbf{x}_i)}{\sum_j q_{\mathcal{F}}^{t}(\mathbf{z}|\mathbf{x}_{j})}\\
q_{\mathcal{L}_{\lambda}}^{t}(\mathbf{z}|\mathbf{x})  & \propto\pi(\mathbf{z})e^{ -H(\mathbf{x}, \mathbf{z}) }  & H(\mathbf{x}, \mathbf{z}) = \boldsymbol{\lambda}^{\top}\mathcal{C}(\mathbf{x}, g_{\mathcal{L}_{\lambda}}^{t - 1}(\mathbf{z}))\\
0  & = \sum_{i}q_{\mathcal{L}_\lambda}^{t}(\mathbf{z}|\mathbf{x}_{i})\boldsymbol{\lambda}^{\top}G(\mathbf{x}_{i}, g_{\mathcal{L}_{\lambda}}^{t}(\mathbf{z})) & G(\mathbf{x}, g(\mathbf{z})) = \frac{ \partial \mathcal{C}(\mathbf{x}, g(\mathbf{z})) }{ \partial g(\mathbf{z}) } 
\end{align}
$$

In the above equation: 
- $\mathcal{L}_{\lambda} = \mathbb{E}_{\rho(\mathbf{x})}[D_{\mathrm{KL}}(q(\mathbf{z} | \mathbf{x}) \mid\mid \pi(\mathbf{z}))] + \boldsymbol{\lambda}^{\top}\mathbb{E}_{\rho(\mathbf{x})q(\mathbf{z}|\mathbf{x})}[\mathcal{C}(\mathbf{z}, g(\mathbf{z}))]$ 
	- This $\mathcal{C}$ is the cost function. The above is the Lagrange equation on the KL between posterior and prior as a bounded optimisation problem, where we ask the constraints to be below 0. 

- The first line comes from setting $\frac{ \partial  }{ \partial g(\mathbf{z}) }\mathcal{F} = 0$
- The second from setting $\frac{ \partial   }{ \partial q(\mathbf{z}| \mathbf{x}) } = 0$ and mean that that the ideal generated is a weighted sum of contributing samples ($\mathbb{E}_{q_{\mathcal{F}}^{t}(\mathbf{x}|\mathbf{z})}[\mathbf{x}]$)

- the rest of the lines come from optimising the Lagrangians, specifically when we are using a Gaussian to compute the likelihood, and setting the global variance $\sigma^{2}$
- $H(\mathbf{x}, \mathbf{z})$ measures toatal energy in the system (Hamiltonian)
- The last line means that the gradient of the cost function is orthogonal to the data manifold. 

# Proposition 2: Equiprobability partitioning of the latent space

Let the training point $x_{i}$ posterior density be:
$$q(\mathbf{z}|\mathbf{x}_{i})=\pi(\mathbf{z})\mathbb{I}_{\mathbf{z}\in\Omega_{i}}/\pi_{i}, \pi_{i}= \mathbb{E}[\mathbb{I}_{\mathbf{z} \in\Omega_i}]$$ 
It is equal to restricting the prior to a region $\Omega_{i}$ and then normalising it. 

The highest ELBO is achieved when the partition is equiprobable under prior: 
$$
\begin{align}

\mathbb{E}_{\pi}[\mathbb{I}_{\mathbf{z} \in \Omega_{i}}]  & = \mathbb{E}_{\pi}[\mathbb{I}_{\mathbf{z} \in \Omega_{j}}] \quad \forall i,j \\
\mathbb{R}^{d_{z}} & = \cup_{i}\Omega_{i} \quad \text{and} \quad \Omega_{i} \cap \Omega_{j} = \emptyset \quad \forall i \neq j
\end{align}
$$

# Proposition 3: Blurred reconstructions

The optimal reconstruction is a weighted sum of the training points. 


# Proposition 4: High capacity $\beta$-VAE and spectral methods

Let $\phi_{a}:\mathbb{R}^{d_z} \to\{ 0, 1 \}$ be an orthogonal basis in the latent space: 
$$
\begin{align}
q(\mathbf{z}|x_{i})  & = \pi(\mathbf{z}) \sum_{a}m_{ia}\phi_{a}(\mathbf{z}) \\
g(\mathbf{z})  & = \psi ^{\top}\phi(\mathbf{z})
\end{align}
$$
$\psi$ are parameters of the decoder, $\sum_{a}m_{ia}\pi_{a} = 1$, where $\pi_{a} = \mathbb{E}_{\pi_{\mathbf{z}}}[\phi_{a}(\mathbf{z})]$

- Then the fixed points equations for ELBO are equivalent to reconstructions of a Kernel-PCA model wiht normalised Gaussian Kernel with scale parameter $\sqrt{ \beta }$

# Proposition 5: Equipartition of energy

*VAE encoders learn a tiling of the lantet space, each tile corresponding to a different level of $H(\mathbf{x}, \mathbf{z})$. This can be a guiding principle to evaluate generative models*

Let $H(\mathbf{x}, \mathbf{z})$ be the "Hamiltonian function" from proposition 1. 

For a given datapoint $\mathbf{x} \in \mathbb{R}^{d_x}$, $\mathbf{z} \in \mathbb{R}^{d_z}$ and some $\epsilon > 0$

define set $\Omega(\mathbf{x}, \mathbf{z}_{0})$: 
$$
\Omega(\mathbf{x}, \mathbf{z}_{0}) = \{ z' \mid H(\mathbf{x}, \mathbf{z}') - H(\mathbf{x}, \mathbf{z}_{0}) \leq \epsilon\} \subset \mathbb{R}^{d_z}
$$
(all latent points where Hamiltonian is constant)

If we vary $\mathbf{x}$ and $\mathbf{z}_{0}$, we find each $\Omega(\mathbf{x}, \mathbf{z}_{0})$ to be an element of a set of *disjoint sets* enumerated by $\Omega_{a}$. The encoder density $q(\mathbf{z}|\mathbf{x}_{i})$ converges to **a mixture of restrictions of the priors to the basis elements $\Omega_{a}$**

Denote $\gamma_{a}$ as the probability of a sample from the prior falling in the set $\Omega_{a}$:

$$
\frac{\sum_{i}e^{ -H_{ia}/\beta }}{\sum_{b}e^{ -H_{ib}/\beta }\gamma_{b}} = n, H_{ia} = H(\mathbf{x}_{i}, \mathbf{z}_{a})
$$

# Algorithm: GECO (Generalized ELBO Constrained Optimization)

t = 0; 
lambda_vec = ones;
while training do
	read minibatch x; 
	sample z from q(z|x);
	compute batch average of constraint C_hat(t) = C(xt, g(zt));
	if t == 0 then
		Initialize constraint moving average C_bar = C_hat(0);
	else
		C_bar(t) = alpha * C_bar(t - 1) + (1 - alpha) * C_hat(t);
	end
	*C(t) = C_hat(t) + StopGradient(C_bar(t) - C_hat(t));*
	compute gradients: G_theta = partial L_lambda / partial theta and G_eta = partial L_lambda / partial eta;
	update parameters: theta = theta - lr_theta * G_theta and eta = eta - lr_eta * G_eta;
	update lagrangian: lambda = lambda * exp(C(t))
	t = t + 1;

Note: 
$$
\begin{align}
\Delta_{\theta, \eta}  & \propto -G_{\theta, \eta} \\
\Delta_{log(\lambda)}  &  \propto C^{t}
\end{align}
$$

Concepts: 
- $\alpha$ is the moving average parameter: to update $\boldsymbol{\lambda}$ we follow a moving average of the constraints. Only the last step of the moving average is applied gradients (as in the italicised line)