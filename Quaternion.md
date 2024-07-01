# 1. from _Quaternion Integration_

## Quaternion Superiority over other reps

Mainly that other representations (Euler angle and RPY) suffer from the Gimbal lock problem where one degree of freedom is collapsed when two axes align. Quaternions do not have this problem. Another pitfall is interpolation ambiguities (rotational matrices suffer from normalisation issues).

## Quaternion Math

### Representation
A quaternion is a 4-tuple $(w, x, y, z)$, where $w$ is the scalar part and $(x, y, z)$ is the vector part.
$$
\begin{align}
q  & = \begin{bmatrix}
w & \mathbf{v}
\end{bmatrix} \\
Q & = \{  \pm 1, \pm i, \pm j, \pm k \}
\end{align}
$$
Two binary operations are defined on quaternions:
Addition: $+$
- this is defined as a component sum; it is commutatiive and associative
Multiplication: $\otimes$
- this is defined as
$$
q_{1} \otimes  q_{2} = \begin{bmatrix}

 w_{1}w_{2} - \mathbf{v}_{1} \cdot \mathbf{v}_{2} & w_{1}\mathbf{v}_{2} + w_{2}\mathbf{v}_{1} + \mathbf{v}_{1}\times \mathbf{v}_{2}\end{bmatrix}
$$
	which is not commutative but distributive over the sum. 
Norm:
$$
\lVert q \rVert = \sqrt{ w^{2} + \lVert \mathbf{v} \rVert ^{2}_{2} }
$$
Conjugate: 
$$
\begin{align}
q^{*}  & = \begin{bmatrix}
w & -\mathbf{v}
\end{bmatrix} \\
q\otimes q^{*}  & = \lVert q \rVert ^{2}
\end{align}
$$
Inverse: 
$$
\begin{align}
q^{-1}  &  = \frac{q^{*}}{\lVert q \rVert ^{2}} \\
q\otimes q^{-1} & = \begin{bmatrix}
1  &  \boldsymbol{0}
\end{bmatrix}
\end{align}
$$

### Rotation

Let's map from complex numbers here. 

In complex numbers
$$
\begin{align}
\hat{k} & = e^{ i \theta} k \\
e^{ i \theta} & = \cos \theta + i \sin \theta 
\end{align}
$$
In quaternions
$$
\begin{align}
\hat{q} & = e^{ \frac{\theta}{2} \mathbf{v}} q e^{ - \frac{\theta}{2}\mathbf{v} }\\
e^{ \frac{\theta}{2}\mathbf{v} }  & = \cos \frac{\theta}{2} + \sin \frac{\theta}{2} \mathbf{v} \\
\end{align}
$$
$\mathbf{v}$ is the axis of rotation consists of i, j, and k directions (right hand rule applies). These directions follow the Hamilton product rule and is noncommutative (reverting the order gives the negative on the RHS):
$$
\begin{align}
i \otimes j & = k \\
j \otimes k & = i \\
k \otimes i & = j \\
\end{align}
$$

Useful link:
[3blue1brown x Eater interactive guide](https://eater.net/quaternions/video/rotation)

Basically, rotation by any unit quaternion (also called a versor) should be thought about as a rotation by an angle perpendicular to any vector axis (defined by $\mathbf{v}$ in $\mathbb{R}^{3}$). The real part of the quaternion is the cosine of the half-angle.

The reason why this is a half angle is that the rotation is defined by the versor $e^{ \frac{\theta}{2}\mathbf{v} }$ and its inverse $e^{ - \frac{\theta}{2}\mathbf{v} }$, which are pre- and post- multiplied to the target quaternion. This way, we undo the warping and preserve the rotation in the desired rotational plane (perpendicular to $\mathbf{v}$).

## Quaternions, Rotation Matrices, and SO(3)

### Group Formulation

Def: a group is a set $G$ along with some *binary* operation (mapping two elements of the set by an operation to give a third)

Rules: 
closure: $a, b \in G \implies a \cdot b \in G$
associativity: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
identity: $\exists e \in G$ such that $e \cdot a = a \cdot e = a$
inverse: $\forall a \in G, \exists a^{-1} \in G$ such that $a \cdot a^{-1} = a^{-1} \cdot a = e$

So 3D rotations is a group! the operation is composition of rotations, the identity is the null rotation, and the inverse is the opposite rotation.

so does quaternions with $\otimes$, same as rotation matrices with matrix multiplication.

Quaternion mulitplication is also a linear operation, as we can write out these multiplications as matrix multiplications. 

$$
\begin{align}
q_{1} \otimes q_{2}  & = \begin{bmatrix}
w_{1}w_{2} - x_{1}x_{2} - y_{1}y_{2} - z_{1}z_{2} \\
w_{1}x_{2} + x_{1}w_{2} + y_{1}z_{2} - z_{1}y_{2} \\
w_{1}y_{2} - x_{1}z_{2} + y_{1}w_{2} + z_{1}x_{2} \\
w_{1}z_{2} + x_{1}y_{2} - y_{1}x_{2} + z_{1}w_{2}
\end{bmatrix} \\
 & = \begin{bmatrix}
w_{2} & -x_{2} & -y_{2} & -z_{2} \\
x_{2} & w_{2} & z_{2} & -y_{2} \\
y_{2} & -z_{2} & w_{2} & x_{2} \\
z_{2} & y_{2} & -x_{2} & w_{2}
\end{bmatrix}\begin{bmatrix}
w_{1} \\
x_{1} \\
y_{1} \\
z_{1}
\end{bmatrix}
\end{align}
$$

The opposite $q_{2} \otimes q_{1}$ can be rewritten analogously.

## Integrating angular velocity

Note the relationship converting an original angle to one after rotation
$$
\mathbf{s} = q \otimes  \mathbf{s}_{0} \otimes q^{*}
$$

Taking derivatives by the product rule (noting the order shouldn't be changed):
$$
\begin{align}
\dot{\mathbf{s}} & = \frac{d}{dt}[q \otimes  \mathbf{s}_{0} \otimes q^{*}] \\
 & = \dot{q}\otimes  \mathbf{s}_{0} \otimes q^{*} + q \otimes  \mathbf{s}_{0} \otimes  \dot{q^{*}}
\end{align}
$$

To solve for $\dot{q^{*}}$, we can look at the conjugate rule: 
$\forall q_{i} \in \mathbb{H}_R, \lVert q_{i} \rVert = 1$ (all unit quaternions)
$$
\begin{align}
\lVert q_i \rVert   & \overset{1}= 1 \\
q_i \otimes q_i^{*} & = 1 \\
\frac{d}{dt}[q_i \otimes q_i^{*}] & = 0 \\
\dot{q_i} \otimes q_i^{*} + q_i \otimes \dot{q_i^{*}} & = 0 \\
\dot{q_i^{*}} & \overset{A}= -q_i^{*} \otimes \dot{q_i} \otimes q_i^{*}
\end{align}
$$
$A$ is for $q_{i}^{*} = q_{i}^{-1}$, provided equality $1$. 

Hence: 
$$
\begin{align}
\dot{\mathbf{s}}  & = \dot{q}\otimes  \mathbf{s}_0 \otimes  q^{*} - q \otimes \mathbf{s}_{0} \otimes q^{*}\otimes \dot{q}\otimes q^{*} \\
 & = \dot{q} \otimes  q^{*} \otimes \mathbf{s} - \mathbf{s} \otimes  \dot{q} \otimes  q^{*} \\
 & = \boldsymbol{\omega} \otimes  \mathbf{s}
\end{align}
$$
Statement: $\dot{q}\otimes q^{*}$ is a pure quaternion (real part 0). Hence, the quaternion commutation (def: $\mathrm{Com}(p, q) = p\otimes q - q \otimes p$) yields productive result: $\mathrm{Com}(p, q) = 2(p\otimes q)$

This can be proved in the following
$$
\begin{align}
\text{purity of } \dot{q}\otimes q^{*}: & \\
\dot{q} \otimes q^{*} + q \otimes \dot{q^{*}}  & = 0 \\
\dot{q} \otimes q^{*} + (\dot{q} \otimes q^{*})^{*}  & = 0 \\
\mathrm{Re}( \dot{q} \otimes q^{*})  & = 0\\
\text{commutation of pure quaternions:} & \\
\mathrm{Com}(p, q) & = \begin{bmatrix}
0 & 2(\mathbf{v}_{1} \times \mathbf{v}_{2})
\end{bmatrix} \\
\text{if } \mathrm{Re}(p)  = \mathrm{Re}(q)& = 0 \\
\mathrm{Com}(p, q)  & = 2(p \times q) = 2(p \otimes q) \\
 \\
\text{Hence:} & \\
\dot{\mathbf{s}}  & = \mathrm{Com}(\mathbf{s}, \dot{q} \otimes  q^{*}) \\
 & = 2(\mathbf{s} \otimes \dot{q} \otimes  q^{*})= 2(\dot{q} \otimes  q^{*} \otimes  \mathbf{s}) = \boldsymbol{\omega} \otimes  \mathbf{s} \\
\dot{q} & = \frac{1}{2}\boldsymbol{\omega} \otimes  q
\end{align}
$$

With the above tool, we can try to solve this differential equation by rewriting it in the matrix form.
$$
\dot{q} = \begin{bmatrix}
\dot{w} \\
\dot{x} \\
\dot{y} \\
\dot{z}
\end{bmatrix} = \frac{1}{2}\begin{bmatrix}
0  & -\omega_{x} & -\omega_{y} & -\omega_{z} \\
\omega_{x} & 0 & \omega_{z} & -\omega_{y} \\
\omega_{y} & -\omega_{z} & 0 & \omega_{x} \\
\omega_{z} & \omega_{y} & -\omega_{x} & 0
\end{bmatrix} \cdot \begin{bmatrix}
w \\
x \\
y \\
z
\end{bmatrix}
$$

Solving this $\dot{q} = \mathbf{A}q$  linear ODE just turns into a matrix exponential: 
$$
q(t) = e^{\mathbf{A}(t - t_{0})}q_0
$$

Here, $\mathbf{A}$ is the big matrix with the $\frac{1}{2}$ scalar intrinsic to it.

We can define quaternion exponentials with the following: 
$$
\begin{align}
\exp(q)  & = e^{ w }e^{ \mathbf{v} } \\
 & = e^{ w }\sum_{k = 0}^\infty \frac{\mathbf{v}^{k}}{k!} \\
 & = e^{ w }\left( \cos\left( \lvert \mathbf{v} \rvert + \frac{\mathbf{v}}{\lvert \mathbf{v} \rvert }\sin \lvert \mathbf{v} \rvert  \right) \right)
\end{align}
$$

With this additional definition,
$$
q(t) = e^{ \mathbf{A}\Delta t}q_0 = \exp\left( \frac{1}{2}\boldsymbol{\omega}\Delta t \right)\otimes q_{0}
$$

Though this is only for if the angular velocity is constant over some time period.

## Author's comments
*However, my description here is far from complete. The equation above only holds if the angular velocity is constant over a time period. This means its a “first order” model. Dropping this assumption gives 𝑛𝑡ℎ order models for integration. There are also intricate details that I’m only beginning to understand. For instance, I’m reading about how rotations are a special type of group called [Lie Groups](https://en.wikipedia.org/wiki/Lie_group) where the group is also a [differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold) (yet another interesting abstract mathematical object). The space of angular velocity forms what is called a [Lie Algebra](https://en.wikipedia.org/wiki/Lie_algebra) on the group. And the quaternion exponential function which most texts I refer to seem to pull out of thin air is actually related to a more general idea called an exponential map which maps general Lie Algebras to Lie Groups. *

# 2. from Lie Groups and Lie Algebra Wiki
https://en.wikipedia.org/wiki/Lie_group
## Lie Groups
nutshell: *smooth differentiable manifolds*

def: manifold
- a topological space that locally resembles Euclidean space near each point

def: group (visited earlier)
- a set with a binary operation that satisfies the group axioms

def: continuous group
- a group that is also a topological space, such that the group operations (multiplication, taking of inverses) are continuous
	continuous symmetry: translations of a function (differential equations to intial conditions) for example; rotation of a sphere. 

def: Lie group
- a group that is also a differentiable manifold, such that the group operations are differentiable

def: Lie algebra
- a *linearised*, local version of a Lie group (Lie himself called the "infinitesimal group"). 


# 3. from Mathemaniac video series
## 1. Prerequisites
- Rotations - higher dimensions or complex?
- Problem: describing rotations in a systematic way, by listing a few properties for vector operations
	1. Linear: $R(u + v) = R(u) + R(v)$ (the rotation of the sum is the sum of rotated components - vector sums; rotations should also be robust to scaling) 
		1. Can write a rotation matrix. 
	2. Preserves lengths and angles (between vectors in the vector field following the transformation): $\mathbf{v} \cdot \mathbf{w} = (R\mathbf{v})\cdot(R\mathbf{w})$; $\mathbf{v}^{\top}\mathbf{w} = (R\mathbf{v})^{\top}(R\mathbf{w}) = \mathbf{v}^{\top}R^{\top}R\mathbf{w}$
		1. $R^{\top}R = I\to R \in O(n)$  (group of orthogonal matrices)
	3. Preserves orientation (a reflection changes it)
		1. $\det(R)= 1 \to R \in SO(n)$ (special orthogonal group)

- For complex rotations, replace $R$ with $U$; $U$ and $\mathbf{v}$ are both complex. 
	1. Linearity: can write $Rotate(\mathbf{v}) = U\mathbf{v}$
	2. Lengths and angles preservation: do $\mathbf{v}^{H}\mathbf{w}$ instead of $\mathbf{v}^{\top}\mathbf{w}$
		1. $U^{H}U = I$: hence want $U$ to be unitary. $U \in U(n)$  
	3. Orientation: $\det(U) = 1$
		1. $U \in SU(n)$ (special unitary group)

## 2. Lie Theory
- Groups: collection of objects with a binary operation that satisfies the group axioms
	1. Closure: $a, b \in G \to a\cdot b \in G$
	2. Associativity: $(a\cdot b)\cdot c = a\cdot (b\cdot c)$
	3. Identity: $\exists e \in G$ such that $e\cdot a = a\cdot e = a$
	4. Inverse (existence of the undoing element): $\forall a \in G, \exists a^{-1} \in G$ such that $a\cdot a^{-1} = a^{-1}\cdot a = e$

- Manifolds: topological spaces that locally resemble (by two-way deformation) Euclidean space near each point
	- SO(3) corresponds to a 3D solid ball (radius (bounded by $\pi$) being the angle of rotation; vector being the axis of rotation). --> 3D manifold, but with "portals" being periodicities. 

- Use tools from group theory + differential geometry to study Lie groups. 

- Tame "curved" manifolds by linear "maps" (atlas): *Lie algebra* (tangent at identity)
	- Exp is the *map from Lie algebra to Lie group*

- Group theory: from $\log(g)$ and $\log(h)$ to $\log(g\cdot h)$;
	- infinite series with bracket operation to rescue. Denote the above respectively as $X, Y, Z$
	- Baker-Campbell-Hausdorff formula: $Z = X + Y + \frac{1}{2}[X, Y] + \frac{1}{12}[X, [X, Y]] - \frac{1}{12}[Y, [X, Y]] + \ldots$
- Lie brackets: 
	1. Closure
	2. Bilinearity
	3. Alternativity: $[X, X] = 0$
	4. Jacobi identity: $[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0$

With Group (lie bracket) and Manifold (Lie algebra), we can study Lie groups in their Euclidean counterpart.

Finite $G$: infinite families + sporadic groups (Mathieu, Conway, Fischer, etc.)
- way fewer Lie algebras $A_{n}, B_{n}, C_{n}, D_{n}, E_{6}, E_{7}, E_{8}, F_{4}, G_{2}$


## Exp

Def: 
1. derivative property: $f'(x) = f(x), f(0) = 1$
modified: let $g(t) = e^{ tx }$
$$
e^{ x } = g(1), \text{where }g(t) \text{ satisfies } g'(t) = xg(t), g(0) = 1
$$
Whole journey: from $g(0) = 1$ to $g(1) = e^{ x }$

Exp map: $exp: \mathfrak{g} \to G$ (Lie algebra to Lie group)
- Note: argument to this $\exp$ can be a vector &c.
- use *parallel transport* to move vectors around the manifold (like a train on a track)
- stop when one unit of time has passed. 

Exp of derivatives
- $\frac{d}{dx}$ is an operator
- $\exp\left( \frac{d}{dx} \right)$ is another operator mapping $f(x)$ to $g(x)$
	starting point: input function$f(x)$; "velocity" being the derivative of the function
	![[Pasted image 20240701100055.png]]
due to the $\frac{ \partial g }{ \partial t } = \frac{ \partial g }{ \partial x }$	condition, we can solve the partial diff equation by "translation"
Hence below is $g(1, x)$: ![[shift operator.png]]
$\exp(\mathbf{a} \cdot \nabla)f(\mathbf{x}) = f(\mathbf{x} + \mathbf{a})$

Can also exp second derivatives $\left( \frac{d^{2}}{dx^{2}} \right)$, or any other operator $\hat{H}$ that is independent of t; --> the formal solution is the solution of the differential equation (for the operator) with initial condition $f(x)$.
