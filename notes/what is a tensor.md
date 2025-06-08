youtube vid: [link](...)

Base filed: $\mathbb{R}$, 
vector spaces: caps ($U, V, W$)
vectors: lowercase ($u, v, w$) (may be indexed)
scalars: lowercase ($a, b, c, x, y, z$) (real numbers for now)

**def** formal product: 

$$
V \star W = \mathrm{span}_{\mathbb{R}}\{ v \star w | v \in V, w \in W \}
$$

example: 

let 
$$
V = \mathbb{R}^{2} = \{\begin{bmatrix}
a \\
b
\end{bmatrix} |a, b \in \mathbb{R}\}
$$
$$
W = \mathbb{R}^{3} = \{\begin{bmatrix}
x \\
y \\
z
\end{bmatrix} |x, y, z \in \mathbb{R}\}
$$

then

$$
V \star W = \mathrm{span}_{\mathbb{R}}\{ \begin{bmatrix}
a \\
b
\end{bmatrix} \star \begin{bmatrix}
x \\
y \\
z
\end{bmatrix} | a, b, x, y, z \in \mathbb{R}\}
$$

this is infinite dimensioned!

example vectors in $V \star W$:

take 
$$
\begin{bmatrix}
2 \\
1
\end{bmatrix}\star \begin{bmatrix}
 5 \\
10 \\
-15
\end{bmatrix} - 3
\begin{bmatrix}
-1 \\
3
\end{bmatrix} \star \begin{bmatrix}
1 \\
0 \\
7
\end{bmatrix}
$$

This is an object in $V \star W$, but no simplification can be taken. (cannot take 5 out; cannot take 3 in)

$$
(cv) \star w \text{ vs } c(v \star w)
$$

and similar stuff: $v\star(cw)$ vs $c(v\star w)$

or distributive rules: $a \star b + c \star b$ vs $(a + c)\star b$

So we want to form a quotient within some subspace of the formal product space. 

$$
I = \mathrm{span} \left\{ \begin{array}
(cv) \star w - c(v * w) \\
v \star (cw) - c(v \star w) \\
(v + v_{2}) \star w - (v \star w + v_{2} \star w) \\
v \star (w + w_{2}) - (v \star w + v \star w_{2}) \\
\end{array}\right.
$$

where

$$
c\in \mathbb{R}, v \in V, w\in W
$$


Now: $V \otimes W = V \star W / I$

where $\otimes$ is the tensor product. 

$\mathrm{dim}(V\otimes W)$ = $dim(V)dimW$ ($\mathrm{dim}(V \oplus W)$ = $dim(V) + dim(W)$)