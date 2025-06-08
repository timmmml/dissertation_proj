# Learning Structures: Predictive representations, replay and generalization

Ida Momennejad, 2020

Message: 

- RL approach to learning representations of the *structure of states*
	- plausible account for human behavior + neural rep of predictive cognitive maps: capturing structures as predictive representations, updated via replays
	- multi-scale successor representations, prioritized replay, policy dependence	

## Background: 

- target task: "navigation of the world", using relational representations as *internal maps for navigation and planning* (Tolman, cognitive maps)
- early theories: these maps are...
	- in Hc (actually, high Hc-cortex interactions)
	- spatial (actually, other state spaces too)
	- Euclidean (path-dependent representations, skewed toward goal locations; sequential trajectories ot goal twist around obstacles)
	- allocentric

- grid fields can capture principle components of state spaces (successor representations) and warping can happen near locations of goals and rewards


Puzzle: *how do brains learn and update cognitive maps and how do they represent and generalize structures?*

- approaches: 
	- manifolds and RNNs (Chaudhuri et al. 2019, Low et al., 2018)
	- relational learning (TEM)
	- topological models (Wu and Foster, 2014, Babichev et al. 2016)
	- reinforcement learning ([[Stachenfeld et al. 2017]])

- this review: representation learning in RL as a focus
	- learn the structure of the state space (spatial locations, experimental stimuli, associated mem items, task states &c.)
		- mappings between perceptual features and state
		- compact reps of relational/associative structures of state space
		- abstractions that enable transfer between tasks and environments
	- multi-step, multi-scale, path-dependent with successor reps and replay

--- 

## Learning Predictive reps of structures

how rep learning and replay help acquire multi-scale predictive cognitive maps

RL 
1. offers testable hypotheses on the *neural implementation*  of structure learning and correspondence to behavior (*Bayesian cognitive models can do structure learning but have limited account on neural implmeentation*);
2. acquires reps of structures in the absence of rewards (canonical example of exploration without reward leads to better performance on tasks with reward)

Classic RL agents: MF and MB

MF: takes actions, caches (discounted expected) values, such as by TD learning. Cheap, fast, but no explicit memory of the relationship among states just policies given states (think controller where $a_{x} = f(x)$, basically)

MB: relational representation of the environment in terms of the transitino matrix T. use this to iterate and compute action policies. though this is very computationally expensive

efficient intermediate: *Successor Representations*
- representation learning and uses abstractions to acquire relationships between a state and all the successor states. 
- *importantly, this representation captures non-adjacent dependencies unlike MB RL's only one-step capture*. 
	- SR + replay: precompute multi-step and multi-scale dependencies

### The successor representation

- dependent on a sacling factor $\gamma$; 
- gradually learn via TD learning of the couns of visits amog states and successors within a given horizon. 
- (successor prediction errors; cache discounted expected future visitations from one state to another state)

count-based compact representation: no storage of transition probabilities; but discounted counts of future visits. 

**Inherent predictivity**

#### Empirical Evidence for the SR

- human behavioral evidence: 
	- Momennejad et al. 2017
		- behavioral experiments with various revaluations to test SR vs. MF/MB/hybrid models. SR + replay (SR-Dyna; an algo that learns SR via direct/replayed experience) captures human performance
	- Botvinick and Weinstein, 2014
		- SR helps discover subgoals for planning in HRL. 

#### Mathematical relations: 

- matrix dissolvant: $R(A, \lambda) = (\lambda I - A)^{-1}$
- fundamental matrix: $N = (I - Q)^{-1}$, where $Q$ is the submatrix of transition probabilities between transient states and this captures the expected time spent in transient states before absorbance 
- communicability distance: $d_{c}(i, j) = \sqrt{ G(i, i)  + G(j, j) - 2G(i, j)}$; $G = e^{ A }$ is the communicatbility matrix. This gives a metric on how information spread between two nodes in a graph
- graph laplacian: $L = D - A$ where $D$ is the degree matrix and $A$ is the adjacency matrix

spectral clustering: 
- eigen-decompose the graph Laplacian to unravel structures of d
- differences
- use top eigenvalues of the Laplacian and cluster nodes in the embedding space

Now, the SR approximates the inverse of the graph laplacian, 

relating back to the Hc: EC grid cells encode a basis on which *functions* in space can be specified, which extracts the *multi-scale* structure of predictive representations for hierarchical planning and subgoal processing (Stachenfeld's dissertation (*learning neural representations that support efficient reinforcement learning*))
- relation to SR: 
	- SR columns (previous occupancy) simulate place fields
	- SR *eigenvectors* look like grid fields

other properties of the place field: 
- asymmetric skew toward goal (explained with SR-Dyna, which learns visitation counts during experience and *prioritizes replay toward goals*, it learns higher future visitations for locations expected to visit more often)
- heterogeneity along the long axis (larger toward anterior Hc)
	- Multi-scale successor representation learned with *different discount parameters* (corresponding to different predictive horizons) can capture this property 

fMRI comparison: 
- use representation similarity hypotesi to compare measured with hypothesized (from let's say different SR scales) 
- [[Brunec and Mmennejad, 2022]]
- (see later sections for deeper analyses in what exactly prefrontal reps are doing (PFC/EC *compute and represent generalization/basis sets* for structures, task sets and schema (for generalization, see XJ Wang's paper))) 

---
## Learning structures and policy-dependence

how predictive reps are learned, role of policy-dependence

Simple generalization of the TD rule affords to learn the SR matrix by experience as follows: 

TD: 
$$
V(s_{t}) = V(s_{t}) + \alpha \left[ r_{t} + \gamma V(s_{t + 1}) - V(s_{t}) \right] 
$$

SR TD: 
$$
M(s_{t}) = M(s_t) + \alpha \left[ \mathbb{1}(s_{t, }s_{t + 1}) + \gamma M(s_{t+ 1})  - M(s_{t})\right]$$

However, one important caveat is that the learning can be policy-dependent or independent, based on how the agent (learns to) samples through the environment 
- random walk gives you the SR from the graph (like what I did); 
	- it is also possible to do direct computation from a transition matrix (neural implementation unknown) $M = (I - \gamma T)^{-1}$
- goal-directed exploration gives biased reps (such as when reward is not uniformly distributed or when there is obstacle). 

**path-dependence**: policy-dependent reps wouldn't point to a goal location *through a wall that obstrructs it*. (Consistent with bats' vectorial hippocampal goal-represnetations, Sarel et al. 2017)

eigenvector-of-SR view: warped grid field for policy-dependent SR (if different from unbiased) - *overrepresentation of often-visited states, consistent with evidence*

## Learning structures via replay and prioritization

how content and prioritization of memory replay update these representations 

replay prioritization can *simulate goal-related states more often than others*
- for example: reward, subgoals/doors, locations on the path to reward --> let's say SR is updated during offline replay, this policy-dependent map emerges. 

characterization of replays: 
- observe during ripples, which occur on the phase structure within theta 
- time-compressed sequential activation of hippocampal neurons, either forwards or backwards. 
	- Ambrose et al. 2016: backward replay (in fact not only different directions but also different cells, as place cells are also direction-tagged). test if changes in reward can affect replay (reverse replay in particular). # of replays +/- propto (relative amount on the track, changes in reward, which are both related to DAergics). 
	- *may suggest reverse replay's role in learning (and NOT forward replay)*

**Dyna**: (Sutton, 1991)
- model-free + model-based components, learned and updated during *direct experience (to update MF + MB) + offline simulated experience (to update MF cached values)*
- Russek, Momennejad, Botvinick, Gershman, Daw (2017): SR-Dyna, where offline replay is used to update the SR in a Dyna-like architecture (which augments SR for performance in transition revaluations)
![[Dyna family.png]]

## Abstractions, generalization, and transfer

computational directions in generalization and transfer