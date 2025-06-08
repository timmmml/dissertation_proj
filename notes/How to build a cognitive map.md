idea: to build a state space of actions such that actions are easily planned and integrated
- 2D space example:
	- state space scheme 1: place cells (where locations are stored as nodes with unique identifiers)
		- planning is hard because you need to search through neighbors
	- state space scheme 2: grid cells (xy coordinates)
		- planning is simple because you can just compute a vector operation ($\mathbf{a} = -\mathbf{x}_{t} +\mathbf{x}_{*}$)

- nonspatial spaces: 
	- spatial learning = understanding relationships on a graph
	- can use the successor representation that is the *sum of n-step transition matirces* $\mathbf{S} = \sum_{n}\gamma^{n}\mathbf{T}^{n}$ (which has boiled in policies) to define the surrounds
		- from this representation, we can calculate the value of all states as $\mathbf{v} = \mathbf{S}\mathbf{r}$
	observation: columns of $\mathbf{S}$ look like place cells (which is the total expected transition from each state), and *eigenvectors of place cell covariance matrices resemble grid cells - SR cells make PCs of the place cells*.

- state inference from sequence learning: 
	- disambiguation of sensory aliased stimuli due to "contexts"
		- different locations, same stim
		- spatial alternation tasks: different task state, same location: lap cells 

	- build de-aliased state spaces by sequential learning. 
		- example: clone structured cognitive graph: lones of each sensory observation that each encodes some history.
			- learn a set of transition weights between clones. 

- path integrating state spaces
	- inferring latent state = *understanding where you are in abstract space*
		- path integration by continuous attractor neural networks which receive velocity input
			- place cells and grid cells ![[Pasted image 20241016134302.png]]
			- and head direction cells (and ring attractors in flies)
		![[Pasted image 20241016134534.png]]
- in these above studeis, recurrent weights are selected not learned (though one could set up path integration as a learning problem via predicting spatial consequences of actions) [[Path integration and CANN]]
![[Pasted image 20241016134849.png]]


Generalization: 
- path integration maps chart abstract relationships. 
- to make sensory predictions, need to know not just senosry knowledge but also how it interacts with real-world representations - by TEM hypotheses, the hippocampus combines abstract (MEC) and sensory (LEC) information and hence generalize existing structures to new learning. 

Composition: 
- generalisation of components of some maps to other. 
- example: 2D navigation
	- underlying 2D geometry
	- "walls and barriers"
these elements can be combined to understand any given task configuration