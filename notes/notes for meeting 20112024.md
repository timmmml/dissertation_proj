past week's results: 

1. experimented with a setup in which I try to train manifold geometry instead of topology
	- velocities on the tangent space
	- integrate to some parametrization of the manifold (x, y; $\theta$)
$$
\begin{align}
\Delta \theta  & = \int \frac{d\theta}{dt} \, dt \\
 & = \int  \frac{d\theta}{d\mathbf{r}} \frac{d\mathbf{r}}{dt} \, dt 
\end{align}
$$
the tangent vector is the derivative of the position wrt $\theta$
$$
T(\theta) = \frac{d\mathbf{r}(\theta)}{d\theta} = \left( \frac{dx}{d\theta}, \frac{dy}{d\theta} \right)
$$
what we have is $v(t)$, which is the velocity that scales the tangent vector. To integrate to $\theta$ from here, we use the above vector as a scaling factor such that 
$$
\frac{d\theta}{dt} = \frac{v(t)}{\lVert T(\theta) \rVert }
$$
Then I did numerical integration
$$
\theta_{t + \Delta t} = \theta_{t} + \left( \frac{v(t)}{\lVert T(\theta_{t}) \rVert } \right)\Delta t
$$
	
- train based on similarity among them. 
- though, I was unable to find geometry signatures
	- for the ring + sine situation, the network successfully reconstructs a ring based on the projected theta. So, at some level (in terms of integration) it seems to take into the geometry into consideration
	

2. identity inputs:
	- it works (as expected, conforming with the position+direction tuning)

3. trained a linear readout network
	- now we can observe the landscape of drifts &c. 
		- for small networks, you find discrete solutions
		- larger networks remedy that

interesting thing to look at the below setup's image
![[Pasted image 20241119223507.png]]
This setup gives some mind-boggling stuff: 