(*This .md jots the specifics for the IIB project*)
# Outline: 
- Timeline for submissions
- Project specifications
- Progress planner (dynamic)

# Timeline
regular meeting time (Michaelmas): 3pm Tuesdays (starting 15/10/2024)

08/10/2024: "Summary for Project Supervisors" and "Michaelmas progress & industry mark form" issued for spervisors

**08/11/2024: (ddl) First progress/industry meeting with supervisor**

**21-29/11/2024: Mini-conferences where students give oral presentations to supervisors and assessors**
- plan backed up by preliminary results

**06/12/2024: (ddl) Second progress/industry meeting with supervisor**

**23/01/2025, 4pm: (ddl) technical milestone report**

**21/02/2025: (ddl) Third P/I meeting with supervisor**

**21/03/2025: (ddl) Fourth P/I meeting with supervisor**

**02/06/2025: (ddl) Final report + technical abstracts; + log book**

**03/06/2025: start of mini conferences**


## Project specs (after Oct 1 meeting w. G)

What we desire to find, following up GRT work on actionability: 
- some network that expresses the following attractor property over SO(3)
- let $f$ denote the network hidden states
- define $\{ \boldsymbol{\omega}^{m}_{t} \}_{t}$ as a particular actuation of a rotation: $\boldsymbol{\omega}_{t}^{m}$ is 3-dimensional angular velocity on time $t$ in trajectory $m$. 
- want the following actionable encoding of the "current" rotational configuration in the below sense

*(redo Dorrell et al. 2023 work on ring/torus &c. with this definition and see about rid cells)*
$$
\begin{align}
\text{let } R_{3}  & = R_{2}R_{1} \\
f(\{ \boldsymbol{\omega}_{t}^{2}\}_{0:T}, f(\{ \boldsymbol{\omega}_{t}^{1} \}_{0:T}))  & = f(\{ \boldsymbol{\omega}_{t}^{3} \}_{T}, 0)\\
\end{align}
$$
*problem: how to constrain initial conditions such that the network starts off on the manifold*
- *could force 0s or learn the initial conditions to always start from, then apply rotations away from there (everythinig is relative thereafter)*
- **easy fix**: *make $R_{1}$ and $R_{2}$ "arbitrary" rotations that cover the entirety of SO(3) - learn from noncommutativity*
- *or compose more rotations *

where $\{ \boldsymbol{\omega}_{t}^{m} \}_{0:T_{R_{1}}}$ integrates to $R_{1}$ and $\{ \boldsymbol{\omega}_{t}^{n} \}_{T_{R_{1}}:(T_{R_{1}} + T_{R_{2}})}$ integrates to $R_{2}$
$f_{\infty}$ stands for the network that encodes the attractor property. 

for now, we can constrain the input angular velocities to what we already have trained: 
- *investigate lqr for quaternions or any optimal control strategy for quaternion-angular velocities*
	- ![[Pasted image 20241004161212.png]]
	- 
- $GRU(q) = \{ \boldsymbol{\omega}_{t} \}_{0:50}$ 
- and we take exact integration to be the true rotations
	- bias in the span of the above?
- maybe just do random sequences that are constrained in some way (saves time and energy). 

After training, we would have some actionable attractor rep by default
- but let's think ahead how we make this result meaningful
	- qualitatively we can check if we retrieve grid cells
	- *how do we metricise our outcome quantitatively? **
		 - autocorrelation-based metrics 
			 - Behrens et al. to assess the hexagonal grids
				 - https://www.science.org/doi/full/10.1126/science.aaf0941
				- ...
	
- relate back to biology: 
	
	- if we find "place cells" --> other notions of actionability, aside from integration-ability of motion signals? (such as compositionality - (PFC with task subspaces))
		- train on rotation composition, tying "R3" to take "R1" and "R2" networks
		- some attractor struction in which some "R3" subspace is the composition of R1 and R2 subspaces
		- suppose a learned representation of R1, but a hyper-network that takes information about R2 to modify network connectivity (by modulation)

	- consider a continuous task involving rotating some object to several targets - the target configs are presented across task phase, each for a few moments. 
		- during presentation the primate must hold the object in the previous configuration
		- or other setup: the idea is to infuse a skill acquisition component where the several rotations form ingredients to a composite skill. 
	- important to represent the current rotation and the target rotation, as well as update the rotation to take efferent copies (rotating in hand-blind condition)
	- maybe we can find that the parietal cortex (or one of the hippocampal areas) contains an actionable rep of the current rotation, the SMA contains an actionable rep of the residual rotation, and M1 contains the effector modules that feeds back efferent copies to both areas.

- *use initial state given by the attractor --> load on some other network which is a "self-consistent" decoder*