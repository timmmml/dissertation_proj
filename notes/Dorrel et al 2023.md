# Dorrell et al. 2023

Problem: want a codebook for some space that can be manipulated reasonably by group operations that represents the transformations described by that space
- example: motion in 2d space

Actionability (3): 

$$
\mathbf{g}(\Delta \mathbf{x} \cdot \mathbf{x}) = T(\Delta \mathbf{x}) \mathbf{g}(\mathbf{x})
$$

Biological (non-negative firing rate + energy constraint) (2): 

$$
\begin{align}
\mathbf{g}(\mathbf{x})  & \geq 0 \\
\int g_{n}^{2}(\mathbf{x})p(\mathbf{x}) \, d\mathbf{x} & = 1 
\end{align}
$$

Functionality (1): 

$$
\mathcal{L} = \int \int \exp\left(  \frac{-\lVert \mathbf{g}(\mathbf{x}) - \mathbf{g}(\mathbf{x}') \rVert^{2}}{2\sigma^{2}}  \right)\chi(\mathbf{x}, \mathbf{x}') p(\mathbf{x})p(\mathbf{x}')\, d\mathbf{x}  \, d\mathbf{x}'
$$
where
$$
\begin{align}
\chi(\mathbf{x}, \mathbf{x}') = 1 - \exp\left( - \frac{\lVert \mathbf{x} - \mathbf{x}' \rVert ^{2}}{2l^{2}} \right)  &  & p(\mathbf{x}) = \exp\left( - \frac{\lVert \mathbf{x} \rVert ^{2}}{2L^{2}} \right)
\end{align}
$$

Here, hence, we can frame this as a constrained optimization problem, where we optimise $\mathcal{L}$ subject to inequality constraints and equality constraints implied in (2) and (3). 


dissecting the functionality constraint a bit down, we see that the loss is high for representing two highly probable points separated by $\sigma$ similarly ($\mathbf{g}(\mathbf{x}) \sim \mathbf{g}(\mathbf{x}')$), but this is counteracted by the $\chi$ term in which if the two points are close in space the loss is low.

parameters introduced: 

table of parameters:

| parameter | description                          |     |
| --------- | ------------------------------------ | --- |
| $\sigma$  | characteristic neural space distance |     |
| $l$       | characteristic spatial distance      |     |
| $L$       | spatial scale for visits             |     |

for 2d torus, we write down the following equation: 

$$
\mathbf{g}(\mathbf{x}) = \mathbf{a}_{0} + \sum_{d=1}^{D} \mathbf{a}_{d} \cos\left( \mathbf{k}_{d} \cdot \mathbf{x} \right) + \mathbf{b}_{d} \sin\left( \mathbf{k}_{d} \cdot \mathbf{x} \right)
$$

notably: frequencies here are 2d 


## Casting this into an optimisation problem: 

- use GRT to write down the form of the code as above.
- optimise parameters therein
	- $\mathbf{a}_{0}, \{ \mathbf{a}_{d}, \mathbf{b}_{d} \}_{d = 1}^{D} \in \mathbb{R}^{n}$
	- $\mathbf{k}_{d} \in \mathbb{Z}$ (or drop the integer constraint if we are talking about large scales)

For large spaces, we drop the integer constraint. 

$$
\mathcal{L} = \mathcal{L}_{\text{functional}} + \lambda_{p} \mathcal{L}_{\text{non-negativity}} + \lambda_{b} \mathcal{L}_{\text{bounded}}
$$

... use GECO to adjust coefs for constraints, such that when the constraints are violated, the coefs increase; vice versa. 


For finite spaces, we require freqs to be integers. 

this comes with difficulty: cannot use gradient-based optimisation. 

but there is a simplification: ensure rep is non-negative and bounded easily as the space is finite. we can analytically calculate the neuron norm and scale params such that it is 1. 

$$
\begin{align}
\lVert g_{n}(\theta) \rVert ^{2} =  & \frac{1}{2\pi}\int _{-\pi}^{\pi}g_{n}^{2}(\theta) \, d\theta = \lVert \mathbf{a}_{0} \rVert ^{2} + \sum_{d=1}^{D} \lVert \mathbf{a}_{d} \rVert ^{2} + \lVert \mathbf{b}_{d} \rVert ^{2} \\
\bar{g}_{n}(\theta)  & = \frac{g_{n}(\theta)}{\lVert g_{n}(\theta) \rVert} 
\end{align}
$$

we can hence only compute the functional and non-negativity losses

We need to also know which freqs are to be included in the code: 

- introduce $D_{\max}$ to bound available frequencies to "can be used"


question for chat
now let's dive in to spherical harmonics and wigner-d matrices.

I am trying to extend some previous work on representations on 2-spheres to 3-spheres. Specifically, I am trying to find some optimal basis functions (neural representations) that represent 3d rotation transformations. let's call the activations g(alpha, beta, gamma) for the population code for a particular rotation. now, I want the following relationship to be true: g(alpha + delta alpha, beta + delta beta, gamma + delta gamma) = T(delta alpha, delta beta, delta gamma)g(alpha, beta, gamma), where T(...) is a matrix. this obviously can be tackled via representation theory such that we can imagine stack some irreps of SO(3) in block diagonal form to matrix G and transform basis by S to recover T. This would introduce limits on the resulting code g(...), similarly as the following case for a code for 2D sphere that uses spherical harmonics on the 2-sphere. Figure out how to extend this to a 3-sphere codebook: