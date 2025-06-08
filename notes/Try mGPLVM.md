Overview of the method: 

$$
\begin{align}
\{ g \}  &  \sim p^{\mathcal{M}}(\{ g \})  \\
f_{i} & \sim \mathcal{GP}(0, k_{i}^{\mathcal{M}}(\cdot, \cdot)) \\
y_{it }\mid g_{t}  & \sim p(y_{it}\mid f_{i}(g_{t}))
\end{align}
$$

Our case:

we have three potential things to represent in total:  
- current angular position
- reference angular position 
- residual motion 

Note in our current implementation, the state space for the controller is a concatenation of the current angular position and a reference position, both as continuous attractor population states. It is possible to have one attractor host the residual movement too, and for the control task the residual motion should be enough.
- we make no distinction between these regimes; 
- all we advocate for is an attractor-controller-internal model circuit to implement control over a manifold. 
- from this model circuit, the defining feature is that the neural population stores and updates an *on-manifold* neural rep of the current state, which is then transformed into a control action. 

Now the goal: to wrap this model with results. 
- if the model is valid, we should find a left-out subpopulation that encodes the current behavioral variable, which is updated by the controller and sensory feedback. 
- within the circuit we should also find exactly another subspace which essentially implements the controller that transforms the current state into a control. (the control subspace). 
- in the two tales (current vs. residual-encoding), the controller can be differnet
	- for if we are only encoding the current behavioral variable, the controller must show joint tuning to current/reference postions (a subset of this type of tuning gives residual motion tuning). 
		- importantly if we analyze the latent variables that the circuit is encoding (let's say the current and the reference), we will see essentially a $T(2)$ situation, but we'd see singular tuning neurons (those that form the attractor) and dual tuned neurons (those that form the controller). 
	- if we are encoding the residual motion in the attractor, this is equivalent to the case where the reference is set at 0. In this case there is only one dimensional structure. 
- we indeed seem to see good orthogonalization in the modular thing but not the 
