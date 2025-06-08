# Representation theory notes from wiki

## Definition: 

Branch of mathematics that studies _abstract algebraic structures_ by representing their _elements_ as **linear transformations of vector spaces**, and studies _modules_ over these abstract algebraic structures.

*algebra*: the study of algebraic structures, which are defined as sets with operations defined on them that satisfy certain properties (groups, rings, fields, modules (generalised vector space in which scalars are on a ring (a field in vector spaces)), vector spaces (a set whose elements can be added and scaled by scalars), lattices (partially ordered set in which *every pair of elements has a unique supremum and an unique infimum*), algebras over a field (vector space with bilinear product), etc.)

makes something abstract concrete by describing elements using matrices and algebraic operations --> simplified calculations &c. 

structures on which this can be done: 
- groups
- associative algebras
- Lie algebras

Example: representation of a group by invertible matrices. 
...

## Action definition: 
 
-  a representation can be defined as an *action* (generalising the linear map action that matrices do to column vectors)

A representation of group $G$ or (associative or Lie) algebra $A$ on a vector space $V$ is a map:
$$
\Phi: G \times V \to V \quad \text{or} \quad \Phi: A \times V \to V
$$

such that the followin map (represented by $\mapsto$) is linear: 
$$
\begin{align}
\Phi(g):  & V \to V \\
 & v \mapsto \Phi(g, v)
\end{align}
$$

let's say $g \cdot v = \Phi(g, v)$, then for $g_{1}, g_{2} \in G$ and $v \in V$. 
$$
\begin{align}
e \cdot v  & = v \\
(g_{1} g_{2}) \cdot v & = g_{1} \cdot (g_{2} \cdot v)
\end{align}
$$
where $e$ is the identity element of group $G$, and $g_{1} g_{2}$ is the group operation of $G$. 

## Mapping definition

omit for now...


## Dorrell et al. supplementary section A: 

Constraining Representations with Representation Theory

let's say $\mathbf{x}$ are elements of a group of interest

*actionability*: matrix implementation of every transformation of the variable
$$
\mathbf{g}(\mathbf{x} + \Delta \mathbf{x}) = \mathbf{T}(\Delta \mathbf{x}) \mathbf{g}(\mathbf{x})
$$

namely, the activity pattern for a shifted input is the same as the activity pattern for the original input, transformed by the consistent matrix mapping $\mathbf{T}$.

The authors here instantiate group elements by this set of matrices $\mathbf{T}(\Delta\mathbf{x})$, and they are to be combined via matrix multiplications, such that $\mathbf{T}(\Delta \mathbf{x}_{1} \cdot \Delta \mathbf{x}_{2}) = \mathbf{T}(\Delta \mathbf{x}_{1})\mathbf{T}(\Delta \mathbf{x}_{2})$

shorthand definition of representation theory (here): studies of sets of matrices that when combined via matmul operations, are isomorphic to a group. 

(broadened definition): represent abstract groups by group of linear transformations of a vector space

Peter-Weyl Theorem: any matrix that is a representation of a compact topological group can be decomposed as a *direct product* of a set of blocks, called "irreducible representaions" (irreps), up to a linear transformation. 

if $\mathbf{T}(\Delta \mathbf{a})$ represents the *the group of transformations of some variable* $\mathbf{a}$

$$
\mathbf{T}(\Delta \mathbf{a}) = \mathbf{S} \begin{bmatrix}
I_{1}(\Delta \mathbf{a})  &  0  & 0  & \dots  & 0 \\
0 & I_{2}(\Delta \mathbf{a}) & 0 & \dots & 0 \\
0 & 0 & I_{3}(\Delta \mathbf{a}) & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & I_{n}(\Delta \mathbf{a})
\end{bmatrix} \mathbf{S}^{-1}
$$

where $I_{d}(\Delta \mathbf{a})$ are the irreps  of the group in question ($\Delta \mathbf{a}$ and compositions thereof). dimensionality is not fixed across these irreps.

An irrep is defined as a representation that cannot be further decomposed into a direct sum of smaller, non-trivial subrepresentations (that the only subrepresentation of an irrep V is {0} and itself)

direct sum of objects: $V_{1} \oplus V_{2} \oplus V_{3}\dots \oplus V_{n}$ is the set of all ordered n-tuples $(v_{1}, v_{2}, v_{3}, \dots, v_{n})$ where $v_{i} \in V_{i}$

as we seek an actionable representation, $\mathbf{T}(\Delta\mathbf{x})$ must fill up the dimensionalityt of the neural space (hence we fill it with irreps and transform with $\mathbf{S}$)