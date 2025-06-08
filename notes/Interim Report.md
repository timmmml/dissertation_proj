(NOTE: this is temporarily written in .md for simplicity, but can be reformatted to latex in due time!)

# Literature Review

## Group Representation Theory 

**def**: a group representation $D = \{ D(X) \}$ of group $\mathcal{G}$ is an *assignment of a non-ingular square $n \times n$ matrix $D(X)$ to each element $X$ belonging to $\mathcal{G}$*

$$
\begin{align}
D(I) & = I_{n}\\
D(X)D(Y) & = D(XY)
\end{align}
$$

the representations multiply the same way as group actions, and the set is the group's image in GL(n). If this homomorphism is bijective (isomorphic), the representation is said to be *faithful*. 

it acts on a column matrix (the *basis vector*) $u$ (whose elements can be anything but needs to be of the same kind, such that one can establish the following group action $X$)

$$
u_i' = Xu_{i}
$$
thus can construct: 

$$
\mathbf{u}' = M(X)\mathbf{u}
$$

however, for convention purposes, we need to write $D(X) = M^{\top}(X)$

### Equivalence 

**def**: two representations are said to be equivalent if they satisfy the *similarity* transformation:

$$
D_{Q}(X) = Q^{-1}D(X)Q
$$

One can check that the group representation pillars above are met. In effect, the linear transformation accounts for a change of basis in the coordinate system (set of *basis functions*). 

$$
\begin{align}
u'  & = D^{\top}(X)u \\
Qu'  & = QD^{\top}(X)u \\
Qu'  & = QD^{\top}(X)Q^{-1}Qu \\
u_{Q} & = D_{Q}(X)u_{Q}
\end{align}
$$

- hence what's described is a linear transformation from the basis vector $u$ to $u_{Q}$

These are the *same* representations


### Reducibility

Two reps can be added by the *direct sum*, which is basically the following relationship: 

$$
D(X) \oplus D'(X) = D(X) \otimes I_{n} + I_{n} \otimes D'(X)
$$
 where $n = n_{1} + n_{2}$. the basis vectors concatenate. 

This larger matrix is in the *block-diagonal* form, and is said to *reduce* into the smaller blocks. 

One may transform a large representation into this block-diagonal form by similarity operations,. When this is no longer possible, the representation is said to be *irreducible*, where blocks each cannot be reduced further. 

One may decompose the large representation as a sum of irreducible representations.

$$
D = \bigoplus_{i}m_{i}D_{i}
$$

When this decomposition is done (expressed by the appropriate linear transformation), we notice the following: 

- components of the basis vector corresponding to the same irreducible representation transform into each other. Those corresponding to $1 \times 1$ blocks are unchanged by group actions. 
	- the story can be put in the language of vector spaces. Assign an n-dimensional vector space $V$ spanned by the basis vectors of the representation, then if ther eis a *proper subspace* $W$ of $V$ (
		if a vector whose column matrix is $w$ belongs to $W$, $D(X)w$ belongs to $W\forall X\in \mathcal{G}$. 
		) 
	- , then we say the representation is reducible. 


### Example: proper subspace of the continuous rotation group:

for the set of rotations about a single axis (1D rotations), let a basis function be:

$$
\begin{bmatrix}
\sin(\theta) \\
\cos(\theta)
\end{bmatrix}
$$

Then the action of a group element $\alpha$ is the following: 

$$
\alpha \cdot \mathbf{u} = \begin{bmatrix}
\cos(\alpha) & -\sin(\alpha) \\
\sin(\alpha) & \cos(\alpha)
\end{bmatrix}\mathbf{u} = \begin{bmatrix}
\sin(\alpha + \theta) \\
\cos(\alpha + \theta)
\end{bmatrix}
$$

This subspace defined by $\theta$ is invariant under the group action and is irreducible.

**The thing extends to 3D rotations for spherical harmonics** $Y_{lm}(\theta, \phi)$

- $\theta$ Is the polar angle and $\phi$ is the azimuthal angle. 
- $l$ is a non-negative integer and $m$ is an integer such that $-l \leq m \leq l$, respectively the "degree" (order of the harmonic) and the "order"

*orbital angular momentum in quantum mechanics*

- spherical harmonics are the eigenfunctions of the angular part of the Laplacian operator in spherical coordinates. (???)
$$
Y_{l}^{m}(\theta,\phi) = (-1)^{m} \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_{l}^{m}(\cos(\theta))e^{im\phi}
$$

P here are the *associated Legendre polynomials* of degree $l$ and order $m$. 

- hence we see that these are but functions on the unit sphere. 

Spherical harmonics look like the following: 
![[Pasted image 20240902104234.png]]
Dg. 1: only real parts are represented! the imaginary part is the same with a phase shift (to be appreciated below)

![[Pasted image 20240902104215.png]]

Dg. 2: the magnitude also plots the radial distance, also the plotted is absolute value squared

Note: take a moment to digest this. When we write the 3D coordinates in polar form, we discover that the magnitude $r$ is always invariant across rotations. Hence, it is appropriate to only define what happens to the $\theta$ and $\phi$ of a particular 3D coordinate. 

### from Dorrell et al: 

seek codes $g(x)$ s.t.

$$
g(x + \Delta x) = T(\Delta x)g(x)
$$

- basically want codes for spatial locations to be a basis function that can be operated on linearly by group action. 
- know the shape of $T(x)$ being equivalent to square the basis function dimension. 
- stack up a set of irreps to fill the dimensionality, then rotate/scale with matrix Q

### translating these constraints on the matrices into those on the neural code: 

example for representing an angle

$$
\begin{align}
g(\theta)  & = T(\theta)(g_{0}) \\
 & = \mathbf{Q}\begin{bmatrix}
I_{1}(\theta)  &  0  & 0  & \dots  & 0\\
0  &  I_{2}(\theta)  & 0  & \dots  & 0\\
0  &  0  & I_{3}(\theta)  & \dots  & 0\\
\vdots  &  \vdots  & \vdots  & \ddots  & \vdots\\
0  &  0  & 0  & \dots  & I_{n}(\theta)
\end{bmatrix}\mathbf{Q}^{-1}g(0)
\end{align}
$$

Here each $I_{i}(\theta)$ is a rotation matrix that acts on the $i$-th irreducible representation of the group, corresponding to frequency $i$; can fit maximally $\frac{N}{2}$ different frequencies where $N$ is the number of neurons

Hence we can just write the following: 

$$
g(\theta) = \mathbf{a}_{0} + \sum_{i=1}^{\frac{N}{2}}\mathbf{a}_{i}\sin(n_{i}\theta) + \mathbf{b}_{i}\cos(n_{i}\theta)
$$
where $n_{i}$ is an integer. 

One dimension must be devoted to the trivial irrep so D can maximally be the largest integer $< \frac{N}{2}$

*In our case, we can figure out the irrep and say the representation is a linear combination of the functions that appear in the irrep of the group in question*

### Orthogonality theorem

**def**: the matrix elements of the irreducible representations of a group are orthogonal to each other.

- if I pick a location on any particular irrep, and concatenate elements for each group element $g$ (finite groups but extends to infinite groups), then 
	1. any two such column matrices coming from different irreps are orthogonal to each other
	2. any two such column matrices from different positions in the same irrep are orthogonal to each other

this adds to the original case that irreps being in the form of orthogonal matrices. 

$$
\sum_{X}\left[ \hat{D}^{\lambda}(X) \right] _{ij}^{*}\left[ \hat{D}^{\mu}(X) \right]_{kl} = \frac{g}{n_{\lambda}}\delta_{ik}\delta_{jl}\delta_{\mu \lambda}
$$

$\lambda$ and $\mu$ are different irreps of $\mathcal{G}$ having dims $n_{\lambda}$ and $n_{\mu}$.

**def**: the characters $\chi(D)$ are defined as the traces of the matrices $D(X)$ where $X$ Is a group element.

we can construct a *character table* to represent the characters of each irrep, for each group element. 

Orthogonality theorem of characters: 

$$
\sum_{X}\chi^{\lambda}(X)\chi^{\mu}(X) = g\delta_{\lambda \mu}
$$

As *group elements belonging to the same class share characters*, one can write the above in the following way (counting across classes, each with $c_{i}$ members)

$$
\sum_{i}c_{i}\chi^{\lambda}(X_{i})\chi^{\mu}(X_{i}) = g\delta_{\lambda \mu}
$$

Using characters we can also write the direct sum in the following manner: 

$$
\begin{align}
D(X)  & = \bigoplus_{\lambda}m_{\lambda}\hat{D}^{\lambda}(X) \\
\chi(X) & = \sum_{\lambda}m_{\lambda}\chi^{\lambda}(X)
\end{align}
$$

## What are irreps of SO(3)? 

two blog-posts: 
1. [Study Irreducible Representations of SU(2) using Fourier Series](https://desvl.xyz/2022/05/08/rep-SU2/)
2. [Irreducible Representations of SO(3) and the Laplacians](https://desvl.xyz/2022/06/16/so3-laplacian/#Determining-All-the-Irreducible-SO-3-modules)

*important note: SU(2) is a double cover of SO(3)!*

$$
\begin{align}
SU(3) & \cong S^{3} \\
SO(3) & \cong \mathbb{R}P^{3} \cong S^{3}/\{ -1, 1 \} \\
SU(2) / \{  -I, I \} & \cong SO(3)\\
\end{align}
$$
### Irreps of SU(2)

define $V_{0}$ and $V_{1}$ as trivial rep on $\mathbb{C}$ and standard rep on $\mathbb{C}^{2}$ respectively. 

- these are irreducible, and we want to extend this family to $V_{n}$ for $n \geq 2$
- Use symmetric products $\mathrm{Sym}^{n}V_{1}$

understand symmetric products as the following: 

Put $V_{n} = \mathrm{Sym}^{n}V_{1}$ (space of homogeneous polynomials of degree $n$ in variables $z_{1}$ and $z_{2}$)

$V_{n}$ has the following canonical basis
$$
P_{k} = z_{1}^{k}z_{2}^{n - k}
$$


define group action for each $g \in SU(2)$

$$
\begin{align}
\rho  & : SU(2) \rightarrow Aut(V_{n}) \\
g  & \mapsto (P(z) \mapsto P(zg))
\end{align}
$$

so $\rho(g)P(z) = P(zg)$ where $z = (z_{1}, z_{2})$ and $zg$ is a matmul.

$$
g = \left(\begin{array}_
\alpha & \beta \\
-\beta^{*} & \alpha^{*}
\end{array}\right),\lvert \alpha \rvert ^{2} + \lvert \beta \rvert ^{2} = 1
$$

Then the action is the following: 

$$
zg = (\alpha z_{1} - \beta^{*}z_{2}, \beta z_{1} + \alpha^{*}z_{2})
$$

and one can write $gP(z) = P(zg)$: P denotes polynomial, and $z$ here is a vector!

$P_{k}(z) = z_{1}^{k} z_{2}^{n - k}$

$gP(z)\in V_{n}$, so $V_{n}$ are $SU(2)$-invariant -> hence we have a **well-defined representation**

- $V_{0} = \mathbb{C}$ is trivial
- $V_{1} = \mathbb{C}^{2}$ is standard (linear maps)

*Proposition*: $V_{n}$ are irreducible

*proof*: Schur's lemma: if $V$ is irreducible and $T: V \rightarrow V$ is a linear map that commutes with the action of the group, then $T$ is a scalar multiple of the identity. 

- this proof essentially follows the two ways to map S(1) into SU(2). (trivial map and the double cover); omitted here

Now, we do diagonalisation of $SU(2)$ itself (the above have been diagonalisation of representations of $SU(2)$)

- pick $g \in SU(2)$. Let 
$$
g \sim \begin{bmatrix}
\lambda  & 0  \\
0 & \lambda^{-1}
\end{bmatrix} \sim \begin{bmatrix}
\lambda^{-1}  &  0 \\
0  &  \lambda
\end{bmatrix}
$$

For SU(2) is unitary, eigenvalues are on the unit circle. $\lambda \in S^{1}$

$$
\begin{align}
g  & \sim e(t) \sim e(-t) \\
e(t)  & = \begin{bmatrix}
e^{ jt } & 0 \\
0 & e^{-jt}
\end{bmatrix}
\end{align}
$$

$e(s) \sim e(t)$ iff $s = \pm t \text{ mod } 2\pi$. Also $e(t)$ is $2\pi$ periodic

let $f : SU(2) \to\mathbb{C}$ be a class function; $f \circ e: \mathbb{R} \to \mathbb{C}$  is a even periodic function (as above per specs of e).

given an even $2\pi$-periodic function $h:\mathbb{R} \to \mathbb{C}$ we can recover it as a class function, by the following: 

define $\Lambda: SU(2)\to S^{1}$, sending $g$ to its non-negative-imaginative eigenvalue,
define $E: SU(2) \to \left[ 0, \pi \right]$ given by $g \mapsto \frac{1}{i}log(\Lambda(g))$

Hence $h \circ E: SU(2) \to \mathbb{C}$ is a class function. 

$h \circ E \circ e(t) = h(t)$ and $f \circ e \circ E(g) = f(g)$

given class functions, we want to know about characters (denote by $\chi_{n}$)

$$
\chi_{n}(e(t)) = \mathrm{Tr}(\rho(e(t))) = \sum_{k = 0}^n e^{ i(n - 2k) t}
$$
by trig, $\chi_{n}(e(t)) = \frac{\sin((n + 1)t)}{\sin (t)} = \kappa_{n}(t)$

$\kappa_{0}(t) = 1$, $\kappa_{n}(t) = \cos (nt) + \kappa_{n - 1}(t)\cos(t)$

spanning the same space as $\{ \cos(nt) \}_{n \geq 0}$, this series is dense in the space of even $2\pi$-periodic functions. These are also linearly independent.

Turns out: 

1. *for continuous class function* $f: SU(2) \to \mathbb{C}$, 
$$
\int _{SU_2}f(x) \, dx  = \frac{1}{\pi}\int _{0}^{\pi}f\circ e(t)\sin ^{2}t \, dt
$$
2. *every irrep of $SU(2)$ is isomorphic to one of the $V_{n}$*

(For proofs, see the blog post)

### SO(3) irreps

using the surjection $\pi : SU(2) \to SO(3)$, we have $\mathrm{ker}\pi = \{ -I, I \}$

letting $W$ be a representation of $SO(3)$, we have

$$
\rho: SO(3)\to GL(W)
$$

Then

$$
\pi*\rho: SU(2) \to GL(W)
$$

define induced representation $\pi * W$ as $g \mapsto \rho(\pi(g))$. If $W$ is irreducible, then $\pi * W$ is also irreducible. We have $\pi * \rho (-I) = \mathrm{id}_{W}$ (note, $\pi * \rho(x)$ evaluates as $\rho(\pi(x))$)

we can thus establish the identity between the irreps of SO(3) and those of SU(2) where $-I$ acts as identity. 

$$
\rho_{n}(-I)P(z) = P(z(-I)) = P(-z) = (-1)^{n}P(z)
$$

Hence, we obtain the following proposition: 
*every irrep of $SO(3)$ is of the form $W_{n} = \pi_{*}V_{2n}$*

blogpost 2 explores the following result: 

let $P_{l}$ be the complex vector space of homogeneous polynoms in three variables of degree $l$, then: 

$$
W_{l} = \{ f\in P_{l}: \Delta f = 0 \}
$$
- harmonic homogeneous polymnomials in $\mathbb{R}^{3}$, which can be also uniquely determined on the unit sphere $S^{2}$ (hint: the previous recorded spherical harmonics)

## But...

- it seems (I don't understand this fully yet) that the basis vectors above are themselves functions, such that group actions merely apply linear transformations on the combination of these functions. 

Take $Y_{1,1}, Y_{1,0}, Y_{1,-1}$: they can each be evaluated on $\theta, \phi$ (see above pictures). 

let's denote a particular rotation by an appropriate quaternion (angle-axis form, ish), $q$

$$
\begin{align}
g(q_{t}) = f_{q_{t}}(\theta, \phi) = \mathbf{Q}\mathbf{D}(q_{t})\mathbf{Q}^{-1}\begin{bmatrix}
Y_{1,1}(\theta, \phi) \\
Y_{1,0}(\theta, \phi) \\
Y_{1,-1}(\theta, \phi)
\end{bmatrix}
\end{align}
$$

we know that the null rep here is $\mathbf{I}$. 

the action of rotation on a rep in this basis is as follows: implement the rotation on the unit 2-sphere and convert to polar and azimuthal angles: 

$$
Y_{l, m}(\theta', \phi') = \sum_{m'}D_{m'm}^{l}(R)Y_{l, m'}(\theta, \phi)
$$
where $D_{m'm}^{l}(R)$ is the Wigner D-matrix

*we effectively spin these functions and let this spin on the function code for the desired rotation!*

how can we imagine neurons doing this? 
*may neurons directly represent extent to which each spherical harmonic is present?*
what about "spin representations"?

summary: 
- Wigner-D matrices are the irreps of SO(3) (and SU(2))
- the form: 
$$
\begin{align}
D_{m', m}^{l}(\alpha, \beta, \gamma) & := \langle l, m'|R(\alpha, \beta, \gamma)|l, m \rangle ] = e^{-im'\alpha}D_{m', m}^{l}(\beta)e^{-im\gamma} \\
d_{m', m}^{l}(\beta) & = \langle l, m'|R(\beta)|l, m \rangle  \\
 & = [(l + m)!(l - m)!(l + m')!(l - m')!]^{1/2}\sum_{k = 0}^{l}(-1)^{l - m - k}\frac{\sin^{2l - m - k}(\beta/2)}{(l - m - k)!(l + m' - k)!(m + k)!(m' + k)!}
\end{align}
$$
so, we may be able to take real parts within to construct basis functions for the neurons. Though, this set of irreps is very different from the others introduced in Dorrell et al. (which have the same dimensionality, only differing in spatial frequency). I emailed William Dorrell to inquire about this. 

Temporary halt on the discussions wrt GT/GRT. 

## by the way: what is a tensor? 

[[what is a tensor]]


## what's going on within mice brains?

source 1: WIlson et al. 2018
- documents behavioural finding on the inertia sensor 
- see [[Literature exploration]]
source 2: Massullo et al. 2019
- genetic perturbation study finding a sheet of barrels. 
- see [[Literature exploration]]
source 3: Gonzalez-Rueda et al. 2024
- sensorimotor transformation in the SC: 
	- not topographic mapping of static spatial receptive fields onto movement endpoints, but
	- kinetic visual featuress (alignment in the vector space between sensory and movement vectors )
	- NN built on this is "ideally placed" to sustain behaviours such as rapid interception of moving and static targets. 
- method: 
	1. dissect microcircuit
		1. trace $\mathrm{Pitx2}^{\mathrm{ON-PRE}}$ (intermediate layers of sc near $\mathrm{Pitx2}$)
		2. find exclusive excitatory retinal drive to $\mathrm{Pitx2}^{\mathrm{ON-PRE}}$ neurons. domains (SZ/SGS, SO/uSGI, lSGI/SGP; arranged superficial to deep)
		3. conclude that one class of premotor neurons receives *direct* retinal inputs (located primarily in superficial layers), the other receives *disynaptic or polysynaptic* retinal inputs (intermediate and deep layers, modulated by a tonic inhibitory gate)
	2. characterise visual stim response properties in premotor neurons
		1. high perportion of neurons in superficial layer have spatially localised receptive fields, and some also have "narrow-field" tuning (direction/orientation sensitivity). Project vertically into intermediate/deep layers. (thus call them with *ssRF* property)
		2. *ssRF* neurons are of a smaller proportion in the intermediate and deep layers. However, most neurons are still tuned to moving gratings, and they observe greater latency and sharp *subtheshold tuning* (may reflect synaptic inputs from the superficial layer). 
		3. preservation of tuning to *kinetic features* across layers (not static images) (externally generated motion-flows (or even internally generated))
	3. characterise visual/motor tuning properties of motor units
		1. used well-defined motor units (tuning to motion vectors in light *and* darkness), observe responses ot moving gratings (not spatially localised)
		- these motor-tuned units are also tuned to moving gratings (within this set of neurons, how many are darkness-motor-tuned?)
		2. infer gaze path resulting from any head movement; compare the inferred gaze path given the deoded head movement (STA per cell) on grating vision tuning per cell
		- find anti-alignment
	- *coherent visuo-motor direction columns: neurons presynaptic to $Pitx2$ tend to cluster in direction coherent columns*.
	- ethologically relevant (chase and catch tasks)

Interestingly, it seems that the mice SC sheet ot barrells use two dimensions to code for RPY: 

![[Pasted image 20240903140230.png]]

It seems from the above that roll looks like yaw x pitch per barrell. From behavioural data: 

![[Pasted image 20240903140518.png]]

Authors claim that there is no covariance between roll and the other dimensions. For low pitch, it seems like roll has larger variance (consistent with the optogenetics).
- however, if the rotation is fully accounted as in Massullo et al. 2019, it should be the case that +ve yaw should relate exclusively to CCW roll. This is not as observed. 

- what looks like the case, though, is that the mice SC motor neurons code the space in terms of target actions (FR gives velocity, not target displacement)

## Human data?

fMRI recordings during mental rotations (such as Therien et al. 2022)?
or EEG in Anomal et al. 2020? 

(there are tons of these projects out there)

# My explorations

## Predictor Architecture

... (RNN going from a prep state to snapshots of the played-out rotation)
200 in the ISN 
## Controller Architecture

Each step: 
$$
\mathbf{h}_{t} = \sigma(\mathbf{W}_{hh}\mathbf{h}_{t - 1} + \mathbf{W}_{xh}\mathbf{x}_{t} + \mathbf{b}_{h})
$$

knobs: 
- nonlinearity: relu vs. id
- full error feedback vs. only difference
- full coordinate vs. camera coordinate

(go to python for this)
![[Pasted image 20240903152715.png]]


observation: basically, we see that it helps a bit (though not too much) to have full error information; taking away the nonlinearity is quite detrimental to the overall performance (see I_out_error). For camera rendering of cube image information, we also see that it helps to have full coordinate information. 

However, performance was pretty good overall. 

(go to python for comparing error types on a globe)


Go to Python to check RDM shapes

Go to Python to check that reps are all similar, in terms of RDMs, for the ranges of rotations concerned in Wilson et al. 2018

Check that when you plot across theta values rotating around the same axis,  you would (in cases where we constrain for stuff, at least), that neurons code axis, and the strength of coding gives you the angle



## Tasks after the meeting

- keep an eye out in case Will Dorrell replies to my email about how exactly one can construct actionable reps in SO(3) in the GRT sense
- the main, very interesting finding seems to be the stuff around 180 degrees (SO(3) vs. regularised quaternions display differences;)   
- it is clear (and proven here) that for behavioural rollout, the double-cover is interesting as it makes sure that both sides of the rotation are taken care of. 
	- can I train a network (or a controller) that can *choose* to do the rotation in the opposite rotation? (is there any advantage to doing so?)

[[what's interesting?]]
 