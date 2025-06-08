Topics covered: 
- previous case: warped space because we are sampling from a distorted space (see investigations with sheets &c.)

- potentially add decay term (such that x(t + 1) = (1 - decay) * x(t) + decay * x(t - 1) + noise))
- set $\gamma = \frac{dt}{\tau}$

- sampling initial conditions
	-  arrive at C from the following equation: 
$$
\begin{align}
\mathbf{A}\mathbf{P} + \mathbf{P}\mathbf{A}^T + \mathbf{I}  & = 0 \\
\mathbf{A}^{\top}\mathbf{Q} + \mathbf{Q}\mathbf{A} + \mathbf{P}  & = 0 \\
\mathbf{C}  & = eig(\mathbf{P})[0:3] \\
\end{align}
$$

- investigate C behaviour

unison stuff 
ssh stuff (now forwarded and set up properly)