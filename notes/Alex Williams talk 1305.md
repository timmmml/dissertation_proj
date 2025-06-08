# Quantifying individuality in neural circuit dynamics

keyword: comparisons 

- across species (Safaie et al. 2023) - record homologous brain regions in the same task 
- across brain areas () *thought to spcialize for different functionalities*
- across individuals *baseline indvidual differences*

Dataset sample: IBL 2021 eLife 
- behavioral variablility in learning dynamics
- novelty: instead of averaging out everything analyze how behavioral var $\sim$ neural var

Example 1 in the talk: compare visual response across 55 brain regions, of mice watching a 53s video sample. 

Method: 
- *distil the dataset into a summary statistics*
	- $55 \times 55$ distance matrix of "finring rate statistics similarity"
	- unsupervised or supervised analyses based on the distance matrix
	- PCA-MDS to analyse the tuning curve space (which is found to recapitulate anatomical similarities)
- *pointer: tuning curve similairty from similarity in distribution of PSTHs*


Example 2: postsubiculum HD tuning Duszkiewicz et al. 2024 Nature Neuro
- different PCA geometries across animals. 
- reliability checked with self (training set vs. test set) distance < different animals (though heavily overlapping)

## Technicalities

### shape distances as metric

measure $K$ networks of each $N$ neurons $M$ conditions. 

- visualize nets in a low-D space
- find cluster of similar networks
- predict behavior from nearest neighbors
- "average geometry across all networks"

Target $K \times K$ distance matrix

#### to define the distance function: 

to define a metric space - a set of elements with a distance metric
$$
\begin{align}
d(x, y) &  = 0 \iff x \sim y   & \text{equivalence}\\
d(x, y)  & = d(y, x)  & \text{symmetry}\\  
d(x, y) + d(y, z)  & \geq d(x, z)  & \text{triangle inequality}
\end{align}
$$

guaranteeing these definitions --> preserves transitive properties for clustering *(Mate asks about the ultra-metric question)*

- from triangular inequality: easy to define a neigborhood around each datapoint so that we can try interpolation &c. 
- "linear regression score" is e.g. not a metric for missing symmetry --> defining metrics is not simple!

Idea: procrustes shape distance
$$
\begin{align}
\min _{Q^{\top}Q = I}  \{ \lVert XQ - Y \rVert  _{F}\} 
\end{align}
$$
- Frobenius distance in the optimally rotated geometries. 
- prove symmetry and triagle inequality easy. 
- caveat: need same neuron number (simple solution by dowining the dimensionality to PCA subspace)

this leads to a familiy of metrics, related by preprocessing transformation on the raw datasets
- say whitening (CCA-based distance) *linear invariance $d(x, y )=0$ if linaer map*

permutation invariance?
rotational invariance by the min operation
linear invariance by preprocessing
nonlinear invariance?

### linear decoding as a framework to interpret these distances

similar geometry = similar decoding
- if the geometry is the same --> similar error patterns from linear decoders. 
- if averaging over decoding class --> similarity across many random projections --> intuition is that the similairty in geometry is retrieved 

