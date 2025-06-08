# Intro
INTRO: the paper outlines the APC (active predictive coding) as an underlying principle of coding in the neocortex. 

c.f. Vermon Newcastle's 1978 claim
c.f. J. Hawkins's *A Thousand Brains*

The APC architecture is adapted here to explain:
1. how we recognize an object using eye movemnets
2. why perception seems stable despite eye movements
	*remark: c.f. J.H.'s various examples on digit exploration of a coffee mug; c.f. hippocampus as a cognitive/relational map*
3. how we learn compositional representations
4. how complex actions (goal-directed) can be planned
5. episodic sensory-motor experiences/abstract concepts

# APC outline

Motivation: studies with neural recordings (Zatka-Haas et al., Steinmetz et al.) show that upcoming movemnts affect activities in the sensory cortices. 

- different action integration across layers: 
	- V1L2/3/5/6 - depolarisation before motor onset
	- L2/3: difference between motor input and bottom-up visual input
	- L5/6: positive integration of visuomotor inputs

Emerging view: 
- *almotst all cortical areas update their representations on the basis of efference copies/corollary discharges of upcoming actions, as well as the results of these actions*

Hypotheses of the APC: 
- there exists a canonical cortical module consisting a *state prediction network*  and an *action-prediction network* within each cortical area. 
	- action is beyond motor commands but also abstract actions (traveling across abstract trees, or various group transformations)
- higher order cortical feedback modulates the dynamics of states and actions in lower areas' state/action networks to inform on the current context

Example connection patterns: 
- V1L5PC -> SC (eye movements, localisation)
- A1L5PC -> IC (localisation)
- S1L5PC -> Spinal Cord (bodily movements) *this is arguably a lot more complex: the spinal cord is linked with a great number of functions*
	- M1L2/3 respond to unexpected visual perturbations
	- M2L2/3 respond to auditory sensation/expectation.

# The APC canonical module

## State-transition function

$$
\begin{align}
s_t  & = f_{s}(a_{t - 1}, s_{t - 1}) \\
\hat{f}_{s}  & \sim f_{s} \\
\hat{s} & \sim s
\end{align}
$$

Learning this creates an internal model of the world (generative model/world model/forward model are names for the same thing). 

- can use this model to unroll and sample potentail consequences

## Policy function

$$
\hat{a}_{t} = \hat{f}_{a}(a_{t - 1}, \hat{s}_{t - 1})
$$
To avoid having to conduct rollouts to solve value maximisation every time, an RL agent can use a policy function that maps state into an action. 

Coupling the policy function and a world model, the agent can correct predictions of world states based on action consequences (and hence form a better state estimation). 

*remark: note the similarity between these concepts and observer/controller pairs in state-space control*
## Implementation

- this involves coupling the functions in laminar structures of a cortical locus. 
- ![[rao box1 c.png]]

State network: 
- key phenom: the cortical function should learn to *anticipate* the sensory consequences of actions and exhibit anticipatory activity before movement (these are importantly very short-term, at least on neural timescales) (*how can more behavioural timescale predictions be made? timing of feedback and difference computation is hard to grasp*)

- anticipatory response example: Audette et al. A1 activity even prior to actions (and there also exhibits suppression for the anticipated sounds) (context: lever-pressing + audio tone).  
- also examples of this anticipation in visual cortex, parietal cortex, and frontal cortex for *eye movemenst*;
- other examples of motion that elicit enticipation = *locomotion*


Action network: 
- the action network is assumed to be a recurrent net where *the outputs encode the dynamics of movement* (initial states pushed onto prior to motion start; activity of motor cortex neurons described by a dynamical system $\dot{a} = f(a, u)$). 
- inputs are not only recurrent but also fedforward (thalamal inputs), providing the current state estimate (confirmed in Sauerbrei et al. that thalamal inputs are critical for dextrous movement pattern generation in mice) - co--modulation of these activities are also seen. 

- (what's the role of the premotor cortex? (pushing M1 to the appropriate initial state?))
	 - higher order states require high processing! *think about this later*
- perception is essentially a function of estimating a set of relevant states given world inputs (this can be of course modulated)


Bringing these large-scale characterisations into a humble cortical column, one may map the "motor cortex" of the column: the recurrent network in layer 5, to $f_{a}$; layer 2/3 receives filtered inputs (as a recurrent hidden layer) maps to the $f_{s}$. 

The output class of L5 can convey information to other cortical areas and the striatum - they may maintain the current state *estimate* $\hat{s}_{t}$: this can be computed by a difference between the outcome of the prediction network and direct sensory inputs (feedforward thalamical). 

L5 motor output neurons send outputs to higher-order thalamic nuclei and receive motor information from SSC &c. Hence can compute prediction error in the space of actions ($\hat{a}_{t} - a_t$) (hence observer for actions)

 - note: actions here don't have to be actual actions - rather, they could be any move along the respective *reference plane* coded by the column. 