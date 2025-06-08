username: tim
password: so3-rok
*need to figure out ssh in pycharm*


## Terminology

$\mathcal{\Phi}$  to denote the space of all possible image inputs, rotations to the same object. Each element we call $\phi$.

$I$ to denote the space of learned representation of $\mathcal{\Phi}$. Let encoding function be $E(\phi) = i$ (potentially some variance term as well?)

- we could learn this type of representations by VAE, or *coevolve this representation function along with the main task*

$K$ be the predictor (forward model)

$C$ be the controller (inverse model)

## Concrete action plan

- goal: tackle the image-to-rotation problem by paired forward-model-inverse-model approach to motor control. 
	- as a free bonus, once this is trained, we may expect the model to have learned a compositional map of the space of SO3, as the full controller has therein learned a map that expresses the transition between any two configurations. 
	- let the imagined rotation be $i'$, the ground truth be $i^{*}$; the residual is going to be $i^{*} \otimes i'^{-1}$. 
	- the success herein will evidence in the model's ability to solve a pair-image task: 
		- given $\phi_{1}, \phi_{2}$ (respectively, from and to)
		- internalise to $i_{1}, i_{2} = E(\phi_{1}), E(\phi_{2})$
		- put the "from" and "to" representations into the controller. $i_{2}$ as reference, $i_{1}$ as initial prediction. 

Network definition: 
One timestep in plan phase:
$$
\begin{align}
i^{*} & = E(\phi_{in}) \\
i'_{0}  & = E(\phi_{0}) \\
\mathbf{u}_t  & = C(i^{*}, i'_{t}) \\
\mathbf{z}_{t}  & = G(\mathbf{u}_{t}, \mathbf{z}_{t - 1}) \\
i_{t}'  & = K(\mathbf{z}_{t}) & t >0
\end{align}
$$

One timestep in movement phase:
$$
\begin{align}
\mathbf{z}_{t}  & = G(0, \mathbf{z}_{t - 1}) \\
\boldsymbol{\omega}_{t}  & = M(\mathbf{z}_{t}) \\
q_t  & = \mathrm{Exp}(\boldsymbol{\omega}_{t})\\
I^{*}  & = q_{T} \otimes  \dots \otimes q_{T_a} \\
\end{align}
$$

Question: should I use the controller in the motion phase? 
- Discussion: 
	- In the motion phase, the reference is no longer there. 
	- if we use the controller, we essentially ask it to perform the inverse operation by removing the reference. 
	- hence, we should not use the controller in the motion phase. 
	- the controller in this case operates only in imagination. 

Let's now detail the different functions involved. 
- $E$: the encoder. This is used to transform the image into a representation. 
	- option 1: use a pre-trained network, produced by VAE or alike.
		- find a chance to implement or borrow from ConvDraw
	- option 2: train this network along with the controller: 
		- reconstruction loss here can be replaced by the forward model's loss. 
		- step 1: train as part of the controller.
		- step 2: train variationally (constrained by the above loss, optimise in KL) to form a better representation that handles uncertainty more elegantly. 

- $C$: the controller: This is the inverse model. 
	- option 1: pretrain this (and optionally the Encoder together) for where perfect forward model is available (takes long to train, but maybe we can shrink the movement phase to 5 steps only to remedy).
	
- $G$: the plant: This is the function to take in the control signal and integrate with its own dynamics to produce movemnet by actuator $M$. 
	- take Guillaume's EI network as a starting point; just constrain the outputs to yield reasonable 


First train the forward model actually (rendering takes very nontrivial time)
- generate a bank of data ("infinite stream"), $\{\mathbf{z}_{0}, \mathbf{I}\}$
	- In using the network: treat output dynamics as acceleration
	- Actuator can be a random linear one - no grad here. 
	- Read some notes on how to discretise the dynamics (the matrix's dynamics is a non-normal ODE - solution requires balance of $\tau$ and $dt$ ($\dot{\mathbf{z}} = \mathbf{W}\mathbf{z}$))  
- train on MSE after the encoding function (for now - later plug this into the main model)