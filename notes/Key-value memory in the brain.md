Gershman, Fiete, and Irie, 2025

Argument: *memory in the brain follows a similar principle as other human-designed information retrieval systems. Posited is a division of labor between a key-storage system in the medial temproal lobe and a value storage system in the neocortex*. 
- accessing the memories: matching queries to each key (much like transformer or hopfield nets), then retrieve *a combination of values weighted by their corresponding matches*
- this is *exactly as*: $\mathbf{Z} = \mathrm{softmax}(\beta\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$

---

## Computational foundations

### From correlation to kernels

Kohonen's correlation matrix memory model (reminiscent of Hopfield original)

- in this paper, lower-cased vars are defaulted to be *row-vectors*

heteroassociation matrix $\mathbf{M}$
$$
\Delta \mathbf{M} \propto \mathbf{k}_{n}^{\top} \mathbf{v}_{n}
$$
neuro interpretation: hebbian learning; $\mathbf{M}_{ij}$ is the synaptic strength between key element $i$ and value element $j$

special case: autoassociative
- then we get hopfield networks
- note when $\mathbf{v}_{n} = \mathbf{k}_{n}W_{v}$, where the key-to-value map is preserved, then the above becomes an auto-retrieval system on the keys followed by a readout on the values


query
$$
\begin{align}
\hat{\mathbf{v}}  & = \mathbf{q}\mathbf{M} \\
 & \propto \sum_{n} \alpha_{n}\mathbf{v}_{n}
\end{align}
$$
where $\alpha_{n} = \sigma(S(\mathbf{K}, \mathbf{q}))_n$ is roughly the similarity between the query and the key $n$. $\sigma$ is the separation operator and $S$ is the similarity kernel
- in the early work, the similarity kernel is that of the inner product
- the separation operator is the identity.

- taking the kernel trick: any PSD kernel can be written as an inner product kernel in some feature space
- RBFs as well (Gaussian kernel then) 

Linear layers too can be seen as key-value memory operators when trained by gradient descent  

$$
\mathbf{W}= \mathbf{W}_{0} + \sum_{n = 1}^{N}\mathbf{x}_{n}^{\top}\mathbf{e}_{n}, \quad \text{where } \mathbf{e}_{n}= -\eta_{n}(\nabla _{y}\mathcal{L})_{n}
$$
this construction is the same as the key-value memory
$$
\mathbf{y} = \mathbf{x}\mathbf{W}_{0} + \sum_{n} \mathbf{x} (\mathbf{x}_{n}^{\top}\mathbf{e}_{n})
$$
essentially, it returns a weighted average of corrections to the initial weight result. 

whatever though - this is a bit of a stretch. 

### Rep structure

how are keys, values, and queries represeneted? 

ML: linear mappings from input vectors to keys, values, and queries...
- inputs could themselves be learned embeddings from raw data

can also do *fixe* key and query mappings
- address space: a "scaffold for indexing information content" 
- hopfield: scaffold is identical to the value space --> content-addressibility (retrieval by content similarity)
- others: random, time-dependent address space

--- 

## Neural substrates: plausible mechanisms of biological implementation 

- rules of learning associations
- rules for storing keys and values

rules of learning:
Tyulmankov et al: three-layer neural network implementation of the key-value memory

layer 1: input $\mathbf{x}$ 
layer 2: activation $\alpha$
layer 3: *retrieved* values $\mathbf{v}$

$\mathbf{K}$ (key matrix) is implemented in the $\mathbf{x} \to \alpha$ weights
$\mathbf{V}$ (disregarding the intrinsic connection to keys for now) is implemented in the $\alpha \to \mathbf{v}$ weights

learning rules
$$
\Delta K_{ij} \propto \mu \gamma_i (x_{j} - K_{ij})
$$
$\mu$ is the modulator (global, say dopamine)
$\gamma_i$ is the local modulator
These factors are designed as binary for now. 


