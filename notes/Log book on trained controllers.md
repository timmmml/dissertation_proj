1D ring: 

02-12-15-10: successful training
- no delay
- no noise
- fed back: pos, vel, accel. 
- internal model has the *past CAN state* and the *current sensory feedback*, as well as the *current motor outputs*
- only predictor and internal model are learned against state prediction error
- pre-determined mapping: rotate the object (xy coord) to the coordinate  corresponding to the path-integrated representation:
	- train readout net on the attractor to predict the position corresponding to path-integrating the attractor's action space.  
	- in theory, alignment means more than this - velocities fed to the attractor could as well be the opposite to velocities fed to the environment. That could be learned with lesser assumption - we start from 0, 0 in exploration time and just explore (let's say randomly)
- *the internal model can rely on a few things*: 
	- just read out the velocity that is provided (except learn the nonlinearity (tanh here))
	- just learn to do corrective action by inferring the action from state_{t - 1}, implied by CAN_{t - 1}, to state_t that's given to it. 
	- just learn the correct forward model from muscle efference
	- or a combination of any of these. 
	
	inferred vs. actual action
![[Pasted image 20250212153119.png]]

analyze contribution of state comparison: 
$$
\begin{align}
\hat{\mathbf{x}}  & = W_{\text{pred}}\mathbf{g} \\
a  & = \arctan_{2}(\mathbf{x}) - \arctan_{2}(\hat{\mathbf{x}}) \\
a  & \approx W_{\text{IM, g}}\mathbf{g} + W_{\text{state}}\mathbf{x}
\end{align}
$$

in the short distance limit between $\mathbf{x}$ nd $\hat{\mathbf{x}}$, this angular distance becomes a directed magnitude of the 2D coordinate difference. However, the direction is not straightforward - it depends on the current placement ot $\hat{\mathbf{x}}$ basically. 

however, it seemed like this is an important ingredient for the 1-layer internal model nonetheless - weird enough. 

one also finds a role of velocity and acceleration. 

sadly no forward model is learned. 
![[Pasted image 20250212160050.png]]


add one delay: 
16-17:
- good performance (comparable to above)

![[Pasted image 20250212162033.png]]
still depends on velocity info

![[Pasted image 20250212162116.png]]

good action
![[Pasted image 20250212162132.png]]

this is due to high correlation between action on every step. 
