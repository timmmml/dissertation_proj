definition: 

neuron activity $\mathbf{g}$ has the following dynamics: 

$$
\tau\frac{d\mathbf{g}}{dt} = -\mathbf{g} + f(\mathbf{W}\mathbf{g} + \mathbf{B}\mathbf{a})
$$
where the steady state is defined by:
$$
\begin{align}
\mathbf{g}  & = f(\mathbf{W}\mathbf{g} + \mathbf{B}\mathbf{a}) \\
\end{align}
$$

where $\mathbf{W}$ is a weight matrix, and $\mathbf{B}$ is an input projection matrix dealing with velocity inputs $\mathbf{a}$ (in the case for path integration; in general, it could be representations of actions on the group in the Lie algebra)

another function: 
$$
\tau \frac{d\mathbf{g}}{dt} = -\mathbf{g} + f(\mathbf{W}_{\mathbf{a}}\mathbf{g})
$$

in this case, the steady state activity pattern becomes: 
$$
\mathbf{g} = f(\mathbf{W}_{\mathbf{a}}\mathbf{g})
$$
- this gets closer to the implementation in Dorrell et al. 2023

to directly calculate the steady state, we may handwavily constrain the dynamics to be linear. Then, 

in scenario 1:
$$
\begin{align}
\mathbf{g}  & = \mathbf{A}\mathbf{g} + \mathbf{B}\mathbf{a} \\
\mathbf{g}  & = (I - \mathbf{A})^{-1}\mathbf{B}\mathbf{a}
\end{align}
$$

while scenario 2 is less straightforwardly calculated. 

However, we can also query the relationship between different representations:
$$
\mathbf{g}(\mathbf{a} + \Delta \mathbf{a}) = f(\mathbf{W})
$$

Anyway, let's dive into early works on head direction cells:
Zhang 1996

underlying: head direction cells in bats

observations: 
- world-centric perspective (compass like)
- landmark-dependent (hence used in conjunction of place/grid fields)

- stable attractor dynamics in neural nets + enlightens *how to shift a stable activity profile*. 
- intrinsically one-dimensional, smooth operation & can analyse by continuous analysis.

principles:
1. must not depend on *body-centric* sensory inputs --> need self-sustaining activities
2. must be derived from *body-centric* sensory input + precise integration of movements

Model: consistent with existent idea about the integration of static and movement informations (current head direction + movement, anticipatory activity in thalamus), but *novel in being an explicit formulation of the dynamics of the HD cell continuum and unifies the static/dynamics mechanisms in terms of **the symmetry of the synaptic weight distribution***

- continuous formulation of the HD cell population firing is defined as: 
$$
\begin{align}
f = \sigma(u)
\end{align}
$$
where $f$ is a scalar-valued function $f(\theta, t)$ that is the average activity of a population of cells *with the same prefered orientation $\theta$*, and $u = u(\theta, t)$ is the average net inputs to these units

define dynamics:
$$
\begin{align}
\tau \frac{ \partial u }{ \partial t }  & = -u + w * f \\
w * f(\theta, t) & =\frac{1}{2\pi}\int _{0}^{2\pi}w(\theta - \phi, t)f(\phi, t) \, d\phi 
\end{align}
$$
by the convolution, we take into account the effect of the neighboring units, where the weight is defined by the distance between the current unit and the surrounding unit. 

note: partial($w$, t = t) is not symmetric in $\theta$, which constrasts hopfield networks where this reciprocal equivalence is hold.

then, we can decompose the weight in the following way: 
$$
w(\theta, t) = w_{\text{even}}(\theta, t) + w_{\text{odd}}(\theta, t)
$$
where $w_{\text{even}}$ is even in $\theta$ and $w_{\text{odd}}$ is odd in $\theta$. 
- the symmetric is assumed to be constant, and the antisymmetric is assumed to *modify rapidly* during a head rotation

Stationary self-sustaining activities, in absence of inputs, pushes the network to some "tuned" activity profile that represents an entity in the group. 
- the zero equilibrium is unstable, any small noise will break the symmetry
- "tuning activity" on the ring functionalizes as that the stable profile can be centred at any unit as a result of the rotaiton-invariance of the system. 
	- analogy: a ball balanced on a *horizontal table* (compared to a stable/unstable equilibrium)

Then we we want to choose an activation function and the synaptic weight distribution 

- traditional case: $\frac{1}{1 + e^{-x}}$
- current case: $\sigma(x) = a\mathrm{ln}^{\beta}(1 + e^{ b(x + c) })$

- the current function works better in the regularization solution, probably because the *traditional sigmoid is symmetric with respect to $180\deg$ rotations around the inflection point*
- the current function can look like a power function for large, positive x's, and $e^{ x }$ for if x is highly negative (from power series). 

If we transform it such that $\sigma(x) = ln(1 + ln(1 + e^{ x }))$ then we can get the positive half to behave like logarithm, preserving the exponential decay on the negative part. 

For distribution of weights, we can just solve the dynamical equation for the steady state: 

$$
\begin{align}
u  & = w * f \\
\bar{u}  & = \bar{w}\bar{f} \text{ (FT)}\\

\end{align}
$$
simple solution of $\bar{w} = \frac{\bar{u}}{\bar{f}}$ may not converge as 
$$
\sum_{n = -\infty}^\infty \left\lvert  \frac{\bar{u}_{n}}{\bar{f}_{n}}  \right\rvert ^{2} < \infty
$$
is not satisfied (for higher frequencies, $\frac{\bar{u}_{n}}{\bar{f}_{n}}$ may have quite high amplitudes) --> continuous solution $w(\theta)$ doesn't exist as we cannot invert this FT. 

Approximate solution: use regularization to minimize the weight L2 norms as an additional term: 

$$
\begin{align}
L  & = \frac{1}{2\pi}\int _{0}^{2\pi}(u - w * f)^{2} \, d\theta + \frac{\lambda}{2\pi}\int _{0}^{2\pi}w^{2} \, d\theta    \\
 & = \sum_{n = -\infty}^ \lvert \bar{u}_{n} - \bar{w}_{n}\bar{f}_{n} \rvert ^{2} + \lambda \sum_{n = -\infty}^\infty \lvert \bar{w}_{n} \rvert ^{2}
\end{align}
$$

This can be minimized simply by taking the derivative wrt $\bar{w}_{n}$
then we can find: 
$$
\bar{w}_{n} = \frac{\bar{u}_{n}\bar{f}_{n}}{\lvert\bar{f}_{n}\rvert^{2} + \lambda}
$$
Here we can use inverse FT. Different $\lambda$ can lead to different behaviours: weak regularization can cause "wiggles"

define (somewhat hand-wavily in the paper) a good shape of the weight distribution to be such that there is one activity peak in the steady state ("place" population code) 
- then we see that a "good" weight distribution to be such that weights are positive for nearby angles and negative for more faraway ones.

- can also use gradient on the loss function

*note that to make the above happen, need to realize that $f = \sigma (u)$*

- investigating the stability yields 
	- the zero equilibrium's stability depends on the nonlinearity
	- if the bias $c$ is small --> the zero equilibrium is stable (and single-peaked states are stable too)
	- else, only single-peaked states are stable

HOWEVER, if we introduce noise in the weight distribution, the *rotation invariance* is destroyed and clustering of peaks happen (found in HC place cell models too (1995)).
- investigate in the literature how this is solved (proposals in the paper: such as synaptic plasticity by Hebbian learning) (but check in the later literature). Sejnowski is the author to look here.


Onto DYNAMIC SHIFTS
The paper largely expanded on the idea of tweaking the assymetric part of the weight distribution to shift the activity peak. In biology, this probably won't be done directly by synaptic plasticity and adjusting weights all the time; instead, it may be the case that there are input nodes within the network whose activities are tuned to internal models dealing with corollatory discharges or sensory inputs, effectively moving the activity peak along the neutral equilibrium surface.


# Khona and Fiete 2022, Nature
title: attractor and integrator networks in the brain
"**insights**" sections are commentary by me, no peer-review and may be incomplete; otherwise, the notes are based on interpretation and summarization of the paper and is supposed to be quite accurate. 

outline: 
1. attractor definition, in and out of noise conditions
2. construction of attractor networks
	1. discrete attractors (memory traces, cognitive states, WM)
	2. continuous attractors (space, head direction, object representation)
	3. anatomical topography and weighth symmetries
	4. *non-stationary* nets
3. neural computations with attractors (mostly examples)
	1. representation and memory (hopfield)
	2. de-noising memories 
	3. robust classification
	4. integration (of paths)
	5. decision making 
	6. sequence generation
4. evidence for attractors in the brain
	1. *criteria to follow*
	2. discrete attractors
		1. up and down states
		2. perceptual bistability
		3. bistability in premotor areas
		4. discrete multistability
	3. continuous attractors
		1. oculomotor integrator
		2. head-direction cells
		3. grid cells
		4. graded WM networks
	4. limit-cycle attractors
5. depatures from attractors
	1. orientation tuning in visual cortex
	2. place cells
	3. motor cortex trajectories
6. overall commentary and look ahead

Goal for reading: 
1. clear, working definition of attractors and how I can build them for my use cases: *use gradient descent + input group self consistency*
2. commentary around them: *definition of success, rigorous reasoning across the work (how to prove that the thing I build via theory can be justified in the brain - with or without data); where the nets are captured in the brain;*
3. forward-looking: ways in which these nets can be interconnected to shape dynamics: *how, for example, a single cortical state-action network implements this;* 
	1. look into ways that integrate internal models and sensory feedbacks. (**extending after the project, immediately**)
	2. look into ways in which these nets can form plan hierarchies. (**for my Gatsby essay, more long-term**)

Hence, reading sequence: 
1. first section on attractor definition and criteria; 
2. continuous attractors, take HD cells and grid cells as two example, read about oculomotor integrators
3. relate to place cells and motor cortical trajectories
4. check flexibility and rigidity commentary
5. glimpse the lookahead to gauge what previous sections are important
- compile notes for immediate interest (present project)
- look into discrete case and other integrators, and jot down extensionary ideas.

significance: 
- formal circuit-level models of brain functions (Hopfield networks), specifying networks in the brain that hold certain variables (discrete or continuous)
	- motor control, sensory amplification/memory, motion integration, evidence integration, decision-making, spatial navigation: *set of states can be stabilized through positive feedback*
	- these models yield predictions on patterns of connectivity/cell activity correlations and could be tested in modern experimental breakthroughs (cell-resolution population data for testing these predictions)
## Attractor definition and criteria

1. define *dynamical system and a set of states*
2. an attractor is the *minimal* set of states to which all nearby states flow eventually
	1. stable fixed point as an example
	2. limit cycle
	3. stable manifold
	4. chaotic attractor
3. in the brain use case, we find attractor systems by identifying self-contained systems and the set of variables from which the dynamics of the systems are derived. 

### Neural states

postulate: the system must be self-contained (the *system definition* includes all the external variables) --> then can assume the system is *autonomous*
- in practice, all brain subcircuits receive inputs from elsewhere in the brain, or the world for that matter
- however, (*simplification 1*) we may define a notion of "effective autonomy" where the inputs are stationary and untuned - no differential drive to the system's attractor states
- *simplification 2*:  we take the weights across the neural network as parameters (thoug in practice they are variables themselves over long timescales) and summarize neural states with firing rates. 
	- to ensure a reasonable DS model of the circuit, we must make sure that these simplifications are valid over oru relevant timescale (of seconds, for example). 
- *summary of assumptions*: ignoring on-time modulations, higher-order molecular/ionic effects related to learning, and say the cellular systmes are governed by spike only. 

more defs: 
- attractor manifold: the set of attractor states, in the case where these attractors trace out an approximately continuous and locally Euclidean. Obviously they can be curved and topologically complex (rings, tori, and spheres &c.)
- states on attractors and be stationary, periodic (limit cycles) or chaotic. 

- a single dynamical system can be consist of *coexisting attractors* that occupy different subspaces of the state space. 
	- the state space is not full! attractors occupy only small subregions
	- lower-dimensional than the state space
	- flow depends on initial condition

**insights**: this connect naturally with the MINT idea from recent developments in Mark Churchland's lab (low trajectory mixing, the state space is largely hollow, and decode motion trajectories based essentially on the attractor manifold it occupies.)
- think about stereotyped motion neural trajectories in a bit - are they attractors? 

### Noise conditions: 

- (as reviewed in Zhang 1996, attractors need to deal with noisy effective weights and have to have their own means of robustness)
	- model perspective noise: not all dynamical variables are captured and all models are false
	- true biological noise: noisy availability for signalling-related proteins, cell internal modulation by cAMP/cGMP balances, &c.  --> variability in effective weights!
	- these can drive shifts in attractor states 
		- we would want the system to be robust to these noises
			- lots of noise-independent observers
			- self-regularization activities by learning rules 
		- self-attracting attractor dynamics
			- signature of attractor networks = *localization of states around low-dimensional subsets in real cases, perturbation-robustness, long-time stability of subset states in autonomy*

## Construction and mechanisms

key principle: 
- use positive feedback to fight against decay of estabilished nontrivial states (such that gradients at the state's proximity point back toward the state)
	- **insight** problem: what about drifts along neutral attractor manifolds? purportedly these carrry meaning, but for these manifolds to be continuous then a small drift against it cannot be hedged against, such that the preserved meaning is not stable and won't be useful.
		- this point to calibration mechanisms

construction: 
- solve the forward problem: given a set of weights, find the attractors. 
	1. can do that by simulation
	2. or Lyapnov function approach for symmetric weight profiles ($W_{ij} = W_{ji}$) + rate models

		**corollary**: the Lyapunov function $V(x)$ for a DS $\dot{x} = f(x)$ is a PD scalar function of the system's state x ($V(x)> 0 \forall x\neq 0, V(0) = 0$); it would have a negative definite derivative along system trajectories ($\dot{V} = \nabla V \cdot f < 0$). The trajectory of the Lyapunov function along system evolution defines the system's evolution based on some energy kernel. 
			- stable attractors = energy minima (unstable = maxima)

- or solve the inverse problem (Zhang's approach, as well as my project): given a set of attractors, find network structure that generates it. 
	- *predictions about the underlying mechanisms*
	- data perspective: *going straight from neural activations, so data-efficient*
	- evolution motivation: the brain has been pressured to solve the inverse problme to *perform computaitons that require attractor dynamics of certain forms* 
	- theoretical neuroscience approaches
symmetries
### Discrete attractors (omited for now)

### Continuous Attractors

- theme: continuous symmetry on the weight matrices (note the close analogy to groups and their actions)
	- principle: pattern formation through Turing instability (local symmetry breaking) and pattern maintenance through global symmetry (such as the rise of single-peaked equilibria from small perturbation to the zero equilibrium in Zhang 1996)
	- this is done through a subset of mutually excitatory neurons that are ON that inhibit all others which inhibit them. 

Conditions (sufficient but not strictly necessary) to provide solutions to the inverse problem for stationary continuous attractors
1. BOUNDEDNESS: nonlinear neurons with saturating responses, or self-inhibiting recurrence with uniform excitatory drive
2. PATTERN FORMATION: strong enough referrence with *competitive* dynamics such that patterns can be formed via Turing instability
3. CONTINUUM: continuum of states --> continuous symmetry in the weighting function (such that across a variable that varies continuously the weights remain invariant) - this could take the form of translational or rotational invariance. 
	- **insight**: this is to ensure that any nontrivial state can be maintained through the system's dynamics

Special case: where we have near-linear or linear dynamics that explicitly uses feedback to cancel decays - this creates a set of linear/planar/hyperplane attractors. 
- here the magnitudes are tuned (in the pattern-forming CANs weight shapes are tuned)

### Non-stationary continuous attractors (omitted for now)

## Neural computations with attractors

### Representation and memory

gist: define representations as assignments of *input states* to *representational states* with retrieval capability
- attractor networks map world states to attractor states and allow storage of this information. 
	- through weights (long-term memory-type storage by learning weights from inputs)
	- through persistent dynamics, given weights (working memory-type storage by maintaining the state in the network)
	- definition of content addressable memory: the ability to retrieve the stored information from partial or noisy cues (as in Hopfield networks)
- prior formation of stable sttes through long-term plasticity is crucial for defining the set of states that short-term memory can maintain 
	- this is true (obviously) for STM models that are basd on persistent activity
	- but also for presynaptic facilitation models - implicit need for LTM to construct reinstatable states and to maintain facilitation states over delays
	- **insight**: *good way to link my Bays project to others!*
- for entirely novel inputs - we will need to be able to decompose it to preexisting "parts" such that attractor mechanisms will work
	
### De-noising memories

Robustness of attractors (at least discrete ones) come from denoising capabilities.
- reconstruction of input cues (wich are potentially noisy/corrupted, or incomplete)
	- **insight**: inputs to cortical regions are often low-dimensional - therefore, highly compressed. Attractor-like dynamics will use each layer's stored memory to drive faithful reconstructions (see hierarchical VAE for concept organization)

- another use of this denoising that occurs everywhere is the restoration of neural activity to low-dimensional, behaviourally relevant manifolds from neural dimension noises ($K\ll N$ situations; regularize the noise in the $N - K$ dimensions to keep the activity on the manifold)

- discrete case: noise is corrected by definition

- continuous/manifold attractors: orthogonal noises are corrected, but drift can happen. Though, this variance is quite small (look in to the literature (ref 15, 45, 55, 56) to if this is the same in nature) ($\mathcal{O}\left( \frac{1}{N} \right)$ is the variance - so if the net is very large then negligible (is it? this is just good reduction, but the rep is low dimensional))
	-  maybe look at reduction by redundant copies and characteristical drives.  this solves the problem by allowing noise-inducted drifts to cancel but signal-related drifts to accumulate. 
	
### Robust classification (omitted for now)

### Integration (with noise)

- timescale boost: from neural to behavioural timescale (10-100 ms --> 1-100 s (1000 fold)). 
- need mechanism to *shift internal states along the attractor in response to inputs that encodes changes in the external variable*
	- copy-and-offset construction: copies of the attractor (with small weight assymetry each) within the network (this is Zhang 1996 idea) --> *velocity inputs* whose components project differentially to the copies can break the dynamic equilibria and shift activity peaks to the most active (appropriately displaced) copy. 
	- total direction/magnitude of this shift == time integral of velocity input to the network --> displacement!

- **insight**: for us, we are trying to build a network based on capacity of this integration on the manifold of our interest.

### Decision-making

- inputs, instead of velocities, are *evidence* for different choices
- then, attractor on the manifold that is the decision variable is shifted by evidence accumulation (gen from 2-arm to n-arm decisions) 
	- there is a selection process that could either be external (decides based on "threshold crossing") or internal (a more complex attractor landscape where the continuous attractor gives way to a discrete one)
		- Neural WTA model (inner-Take-All) is a good example of the latter
		- balance between integration and competition 1. may change across decision making (timed modulation or value dependence (define a flowfield)) and 2. affects reversibility of the decision. in the latter setup (flowfield), there are points within the flowfield that respect integration of evidence, others that respect competition. 
	- **insight**: this is a good way to think about decision-making in the context of RT-RNN project and the multitask project.
	
example: Kriener, B., Chaudhuri, R. & Fiete, I. Robust parallel decision-making in neural circuits with nonlinear inhibition. _Proc. Natl Acad. Sci. USA_ **117**, 25505–25516 (2020).
example 2: Wong, K.-F. & Wang, X.-J. A recurrent network mechanism of time integration in perceptual decisions. _J. Neurosci._ **26**, 1314–1328 (2006).
example 3: Prat-Ortega, G., Wimmer, K., Roxin, A. & de la Rocha, J. Flexible categorization in perceptual decision making. _Nat. Commun._ **12**, 1–15 (2021).

### Sequence generation

- sequential behaviour: 
	- construct sequences as low-dimensional limit-cycle attractors: correction + systematic/periodic flow
	- on-going de-noising to prevent dispersion of activity in the sequence-exacting attractor. 
- it is not possible, as in the case for the stationary case for manifold attractors, to correct noise "along" the manifold
	- observation (which follows this theory) is that noise in the timing for obtaining the T-th element scales up with $\sqrt{ T }$.
	
## Evidence
- ...


# Sandro Romani's talk 2019

- 2D HD coding: see activity bumps with heading informtation (actually 1D)
- investigate bat HD cells wrt. 3D rotations 

finkelstein et al. 2015: conjunctive coding for 3 degrees of freedom (Euler angles)

fly: 1 degree of freedom (yaw); classic WTA activity bump model
2 df: 3D heading (yaw and pitch, ignore roll) 
- example rep: project to an 2D plane
	- problem: some smooth transition in head rotations map to abrupt transitions in the map. 
- solution: use the 2-sphere
3 df: similarly to the 2D case, we want to really respect the structure of the space. 

SO(3) 
- 3D sphere, when projected onto a 3D globe, has antipodal points that are the same.
- idea: decompose the "tuning function" into Fourier modes for the manifold of interest

Fourier modes for 1d are $\{ e^{ ik\theta } \}_k$
- $e^{ ik\theta }\cdot e^{ ik\theta' } = e^{ ik(\theta + \theta') }$

- key point: we need to respect composition relationships in the group
- this strategy generalizes to compact Lie groups --> then we can study network solutions

Representation of these groups -> construct the above function!

basis function will be spherical harmonics here. 
$f(g) = a_{0}R_{0}(g) + \mathrm{Tr}(\mathbf{A}_{1}\mathbf{R}_{1}(g)) + \mathrm{Tr}(\mathbf{A}_{2}\mathbf{R}_{2}(g)) \dots$ 
$\mathbf{R}_{1}$ is the first Wigner-D matrix, etc.

the matrices ($a_{0}, \mathbf{A}_{1}$ in their work) want to be symmetric
- 7 free parameters

One solution: 
- systematic way to consider the group manifold and generate excitation local couplings --> solve for dynamical equations wrt. weight distributions, functions of coefficients up to the first mode.

- result: see "bumps" - they can sit anywhere
- head orientation coding?

![[Pasted image 20241019195439.png]]

the hope is that the activity patterns can explain the bat cells activity patterns. 
![[Pasted image 20241019195845.png]]
- idea: thre are pure cells too! and there is an assymetry of cells caring about different dimensions: more cells care about azimuth (> others)

- why different networks in the same animal? (granted that the fourier models can )
	- perhaps the system is performing some computation
	- the computation may be just integration of angular velocities!

Skaggs et al. 1995: consider additional rings that code for self motion. 
![[Pasted image 20241019200304.png]]
- set connections in a proper way --> linear integrator

for 3D rotations: we just need to have additional networks that takes on all the group generators!
![[Pasted image 20241019200432.png]]

orientations of object not self: 
- bit about mental rotation experiment

cool thing: RT correlates with mental imagination of rotation. (corresponding with the "perspective")
- same relationship with "different depth" pairs.

model of the computation: 
- one attractor
- "show image" = bias a subset of neurons that code for the orientation
	--> generates a bump
	- then show another stimulu to bias the network to another bump
	- then, the goal is to move this bump to the other bump
	- the network discovers orientation automatically. 
- time taken for the network to do the thing prop to the angle between the two images. 

- other groups: projective transformation (imagine what the image would look like from another PoV)
