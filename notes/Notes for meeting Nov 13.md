Outline:
1. bumps on the manifolds, and shifts along it.
2. analysis of weights 
3. regarding the zero-input situation

## 0. recap

What I have done: 
- designed another data-picking method: this time we no longer rely on sorting, but can follow any distance function between trajectories!
- no longer need to normalize - training converges as long as the nonlinearity is sigmoidal
- tried zeroing the initial state
	- cannot zero everything, or there won't be any activity!
	- biases, nonzero inputs, noises
	- for tanh, we can do a reasonable job on the ring without noise (we at least to do curriculum learning - very hard to train with noise from the beginning)
	- ![[Pasted image 20241112225246.png]]here all we have is an initial condition on the input nets;  
	- seems to work for SO3 too (somehow sometimes I can train what I think to be SO(3) just from 2 input channels), but when I actually train on 3 input channels it doesnt converge again. puzzling
		- *turns out SO3 looks a bit different still*
	- 	![[Pasted image 20241112230048.png]]
Turns out that tanh seems the easiest to train to do SO3

One particular initialization of the bounded RELU gives us the suboptimal solution as follows (none of any other succeeds at all)
![[Pasted image 20241112233141.png]]

The thing is weirdly easy to train with noise with dim == 2, though, even when I constrain the initial condition to be 0 for both input and rep. Though zero has nothing to do with the null rotation, similar to the case with the ring with this activation. 
![[Pasted image 20241112233529.png]]![[Pasted image 20241112234040.png]]
- we can also use the exact same setup as above, but still the zero has nothing to do with the null rotation 


## 1. bumps on the manifolds

see vids

## 2. weight analysis



## 3. zero initial condition and setups for the next step