Exposition on a continuous valued hopfield network. 

Definition: 

keys: 
$$
\begin{align}
\mathbf{x}_{i}  & \in \mathbb{R}^{d}, \quad \mathbf{X} = \{ \mathbf{x}_{i} \}_{i \in [1, N]} \\
M  & =\max_{i}\lVert \mathbf{x}_{i} \rVert 
\end{align}
$$
queries: 
$$
\begin{align}
\boldsymbol{\xi}  & \in \mathbb{R}^{d} \\
\mathrm{lse}(\beta, \cdot)  & = \beta^{-1} \log \sum_{i} \exp(\beta \cdot_{i}) \\
\end{align}
$$

energy function: 

$$
\begin{align}
E & = -\mathrm{lse}(\beta, \mathbf{X}^{\top} \boldsymbol{\xi}) + \frac{1}{2} \boldsymbol{\xi}^{\top} \boldsymbol{\xi} + \beta^{-1} \log N + \frac{1}{2}M^{2}
\end{align}
$$

*update equation:*
$$
\boldsymbol{\xi}^{new} = \mathbf{X} \mathrm{softmax}(\beta \mathbf{X}^{\top} \boldsymbol{\xi})  = f(\boldsymbol{\xi})
$$

Theorem 1: 
*the update rule converges globally*: for $\boldsymbol{\xi}^{t + 1} = f(\boldsymbol{\xi}^{t})$, $E(\boldsymbol{\xi}^{t}) \to E(\boldsymbol{\xi}^{ *})$ as $t \to \infty$ for a fixed point $\boldsymbol{\xi}^{*}$

Theorem 2: 
*from the iteration rule, in the limit of $t$, $\lVert \boldsymbol{\xi}^{ t + 1}  - \boldsymbol{\xi}^{t}\rVert \to 0$*. Either convergence or reaching a member of the limiting set $\mathcal{L}(E^{*})$, where $\mathcal{L}(a) = \left\{ \boldsymbol{\xi} \in \mathcal{L} \mid E(\boldsymbol{\xi}) = a\right\}$ and $\mathcal{L}$ is the set of stationary points of the update rule. If $\mathcal{L}(E^{*})$ is finite: $\left\{ \boldsymbol{\xi}^{t} \right\}_{t = 0}^{\infty}$ converges to some $\boldsymbol{\xi}^{ *}$
- either convergence or a connected and compact set. 

Theroem 3:
exponential storage. predifining failure probability $p$: 
$$
N \geq \sqrt{ p }c^{(d - 1)/4}
$$
Theorem 4: 
*with query $\boldsymbol{\xi}$, after one update, the distance remaining between $\boldsymbol{\xi}^{new}$ and $\mathbf{x}_{i}^{*}$ (for one $i$) is eponentially small in the separation $\Delta_{i}$.*
where
$$
\Delta_{i} = \min_{j, j\neq i} (x_{i}^{\top}x_i - x_{i}^{\top}x_{j}) = \lVert x_{i} \rVert^{2} - \max_{j} \mathbf{x}_{i}^{\top}\mathbf{x}_{j}
$$

Theorem 5: 
*exponentially small retrieval error*
$$
\lVert f(\boldsymbol{\xi}) - \mathbf{x}_{i} \rVert  < = C e^{ -\beta \Delta_{i} }
$$

transformer relationship: 

for a transformer such as in language models, the update equation given query $\mathbf{Q}$ (let's say current token processed with $W_{q}$) , key $\mathbf{K}$ (let's say context processed with a $W_k$), and value $\mathbf{V}$ (readout processed with $W_v$) is:
$$
\mathbf{Z} =  \mathrm{softmax}(\beta \mathbf{Q}\mathbf{K}^{\top})\mathbf{V}
$$

$\mathbf{Q} \in \mathbb{R}^{N \times d}$, $\mathbf{K} \in \mathbb{R}^{N \times d}$, $\mathbf{V} \in \mathbb{R}^{N \times v}$, and $\mathbf{Z} \in \mathbb{R}^{N \times v}$

this relates to the one-step update of the hopfield network. 

$$
\begin{align}
\mathbf{X}^{\top} = \mathbf{K} = \mathbf{Y}W_{k} \quad & \text{and} \quad \boldsymbol{\Xi}^{\top} = \mathbf{Q} = \mathbf{R}W_{q}\\ 
\mathbf{V} = \mathbf{K}W_{v} = \mathbf{X}^{\top}W_{v}
\end{align}
$$
$$
\begin{align}
\boldsymbol{\Xi}^{new}  & = \mathbf{X} \mathrm{softmax}(\beta \mathbf{X}^{\top} \boldsymbol{\Xi})\\
\mathbf{Q}^{new}  & = \mathrm{softmax}(\beta \mathbf{Q}\mathbf{K}^{\top})\mathbf{K}
\end{align}
$$

so this is to say: transform each word into the query space, retrieve the most similar memory in the key feature space, read out the keys in the value space. This could give you loads of freedom in the power of the update, as the transformation now maps to a set of learned features. 

skipping the several architectures for now