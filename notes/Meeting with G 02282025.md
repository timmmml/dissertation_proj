Problem identification: 
- why do we not achieve perfect velocity prediction, even though the transfer function from motor efference ($a$) to actual angular velocities is extremely simple ($\omega = W_a a$)?
- even the simplest internal model can work (at least when there is no disturbance to correct). 
- despite that, if we set
$$
\hat{\omega} = W \begin{bmatrix}
g^{\top} & s^{\top} & a^{\top}
\end{bmatrix}^{\top}
$$
- we learn some whacky curves
	- plot 1: full usage of all inputs, big W to train; achievement after 300 epochs on prediction: 0.04~0.05
		![[Pasted image 20250227202703.png|300]]		![[Pasted image 20250227205445.png|300]]		![[Pasted image 20250227202735.png|300]]
		
	- plot 2: only use $a$, small W to train; achievement after 300 epochs on prediction: 0.045~0.05
		![[Pasted image 20250227203506.png|300]]![[Pasted image 20250227203530.png|300]]![[Pasted image 20250227203539.png|300]]
Notice the difference? 
- similar performance overall, the restricted model is much more spot-on with the velocity predictions (yes, I made sure the internal model can achieve it, with low efforts, by moving away the nonlinearity and leaving one layer only in the loop).
- also, if you look at the prediction performance plotted across time, you notice a drift away from perfect state recovery. This is because our CAN is sadly not perfect. It is not crystal obvious here, but the weird velocity mapping with additional information to the internal model may represent the effort the model pays to learn to correct some of the accumulated drifts. 
To be clearer, here is a decomposition of the situation.
![[Pasted image 20250227205629.png|800]]

Then, we can compare performances of a more complex internal model: a simple relu MLP with one hidden layer:
- full usage: performing 0.02~0.04 error
- ![[Pasted image 20250227210038.png|300]] ![[Pasted image 20250227210056.png|300]] ![[Pasted image 20250227210123.png|900]]
The action-only thing: 0.034-0.037 steady
![[Pasted image 20250227224129.png|300]] ![[Pasted image 20250227224145.png|300]] ![[Pasted image 20250227224231.png|900]]  

Looks like the performance is better with full information. 
Of course, there is no hope for the action-only thing to ever correct any drift/disturbances. 
Can the full information model correct disturbance?

Well, not really, sadly. 
![[Pasted image 20250227225145.png|300]]

if trained with some noise in the motor efference, the thing becomes slightly better:
![[Pasted image 20250227225601.png|300]]
indeed, this is seen in the following: 
![[Pasted image 20250227231132.png]]

after fine tuning:
![[Pasted image 20250227231956.png]]
![[Pasted image 20250227232031.png|900]]
slightly better? 

Hooking in the controller doesn't take away our perturbation correction ability:
![[Pasted image 20250227234243.png]]
![[Pasted image 20250227234205.png]]
![[Pasted image 20250227234223.png]]

![[Pasted image 20250227234258.png]]
(notice that I have changed the color map to coolwarm here and plotted stuff separately. this is best because it conveys information about our time in trial as well as the position of the disturbance (I put it in the middle!))
The reason why y is more disturbed than x is trigonometry (or, that we are confined on the circle)

If you set everything trainable (without freezing anything), you destroy the previously trained stuff. 
![[Pasted image 20250227234852.png]]

somehow, when you train everything together but with the prediction loss still plugged in, then you get much better recovery
![[Pasted image 20250227235200.png]]
![[Pasted image 20250227235235.png]]
control is slightly worse, though
![[Pasted image 20250227235318.png]]

One can then tune the weighing parameters for best control of operator position. 
For example, the random choice of a parameter here yields an average 0.16 distance from (0, 1). 

But when we froze the stuff, we got 0.11

A better idea is to use stop gradient. Let's get two optimizers for the controller and the internal model-predictor-loop. and segment the loss from which we run backward.

Do it tomorrow. 