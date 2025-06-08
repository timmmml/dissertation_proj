# Michaelmas presentation outline

## Problem motivation 

Big question: how does the brain (neural circuits) hold stable representations of continuous state variables?
- in RL, state variables are important to define the agent's policy functions on a vast number of scales
	- if I am in state x, where should I go next? 
$$
\pi(\mathbf{x}) = \mathbf{a}
$$

- each of these variables live on different (possibly nonlinear) manifolds. examples: rings, tori, &c. 
- if I am motor cortex of a macaque in Byron Yu's lab (c.f. 4G10 L9 BCI learning), my policy function would operate in 2D space and basically say the following (up to the internal model)
$$
\mathbf{u}_{t} = \pi(\mathbf{x}_{t}') = f_{\mathbf{u}}(\mathbf{x}^{\star} - \mathbf{x}_{t}')
$$

- where the $\mathbf{x}'$  is what the internal model of the agent propose its current location is. 
- the policy function here is "neural activity such that my velocity points to my target"

$$
\mathbf{x}_{t}' = f(\mathbf{x}_{t - \tau}^{s}, \{ \mathbf{u}_{t -i} \}_{i \in [0, t - \tau]})
$$

- hence, there should be somewhere in the brain that holds a representation of this current state estimate, which can update via internal model by corollary discharges or sensory inputs. 
*sensory information is partial and delayed*

It is interesting to investigate brain representations of these states, which are not necessarily Euclidean too. 
- the attractor network offers mechanistic insights into how these states are represented in the brain, in providing insights in how they look like and how representations interact with inputs.

Success story about the ring attractor in Drosophilla. In that case this is explicit. 

More up-to-date literature began looking at these in say grid cells in the hippocampus. 

However, a gap is that these networks are usually hand-designed (e.g. by solving equilibrium equations) or trained by supervised learning (by predicting recorded neural activity or 2D coordinates). This becomes hard when we try to march into realms of higher dimensional spaces.

*with supervised learning you assume some decoder and you implicitly lay some constraints on the manifold representation/dynamics*

The project aims to bridge this by training RNNs on contrastive losses, which exploits the relationship between state representations and the space of input channels, such that the states for the set of inputs that lead to similar locations on the manifold are nearer to each other than to other places. 

1D ring: input is angular velocity
2D torus: input is x and y angular velocities
3D torus: input is x, y, z angular velocities
SO3: input is also x, y, z angular velocities
S^2: input is x, y angular velocities

*integration of angular velocities in SC*

Requirement: 
- a queryable function that captures some natural similarity between states (such as geodesic distance)
- no more need to manually force some rep onto the network

## Methodology

Partitioned RNNs 
- rep net to hold the represenation
- input nets to hold the tangent space. 
- inputs are added as a gain to the input->rep communications. 

Contrastive Loss: 
$$
- \log \frac{\exp\left( \frac{s(\mathbf{x}_{i}, \mathbf{x}_{j})}{r} \right)}{\sum_{k} \exp\left( \frac{s(\mathbf{x}_{i}, \mathbf{x}_{k})}{r} \right)}
$$

Activations: 
- ReLU (must pair with L2 loss to prevent crazy activations))
- tanh
- BiologicalReLU

Input generation: 
AR random walks in each channel
- optional variance enveloping

## Experiments and Results

### Phase 1: 

Question: what do the representations look like when trained on contrastive losses? 

underlying:
- symmetric manifolds: (1D ring, 2D torus, SO(3), S^2)
- proof of concept on more bizarre manifolds: (sin x 1D ring, bunny of S^2)

analysis with different constraints: 
- assigning null state to the zero representation
- using identity input-to-rep weights
- using identity rep-to-rep weights
- discrete vs. continuous time solutions

weight distribution comparison with estalished results
- ring
- (very large) 2D torus

aligning hyperparameters to other models 

*Challenge*: design and train this for chimerical manifolds
- large 2D torus x SO(2) --> 2D object manipulation
- large 3D torus x SO(3) --> 3D object manipulation

### Phase 2: 

Question: how do these representations support motor control? 

implementation of various computations: 
- inversion
- composition

implentation of feedback/predictive control:
- generate motor signals as a function of an object on one of these manifold

*Challenge and PhD direction*: how does hierarchical motor control load onto this framework

### Phase 3 (in the unlikely event that I have extra time, or later in my life): 

*this is well worth another project*
Question: there are stabilized machinery to robustly implement these reps for the self states (ring attractor in drosophila, but also entorhinal cortex). How do these representations underlie state representations that appear to be built and used on demand in other parts of the cortex? 

examples: object-oriented representations, chained representations of limbs in motor control &c. 


