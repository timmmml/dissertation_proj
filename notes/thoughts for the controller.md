## Simple PID control\

1. Random readout matrix + simple PID
	- start with parameters $k_{i}, k_{p}$ only; manual design
	- then go to optimised $k_{i}, k_{d}$ by back propagation
	- try derivative control if deemed necessary

2. Optimise the readout matrix $\mathbf{B}$; 
	- do so with fixed $k_{i}, k_{p}$ first; then can jointly optimise \
	- $\mathbf{B}\in\mathbb{R}^{200\times 16}$ (let's deal with the `camera_simple` design first)

3. Use RNN to function as a controller
	- compare performance of RNN controller (plugged after the predictor) vs. other designs.

We can end up with multiple working "representations" of SO3.
- define such representation as a *bijective* mapping between the space of SO3 and some preparatory state space (200 dimensions for the ISN network). 
- this preparatory state can be reached in a few different ways: 
	1. (current): predictor + feedback controller (compare simple with RNN controller and see if reps are similar across controller designs that work). 
	2. (tried but must try again with simpler mesh objects): Feedforward controller (maybe RNN structure - directly learned end-to-end to *find* a mapping between configuration and initial condition)
		 - can use the predictive model to train this! ()
	3. feedforward end-to-end

## iLQR (further down the line, if the above don't work.)
$$
\begin{align}
\mathbf{I}(t) & = \mathbf{u}(t) + \epsilon_t\\
\text{let } \epsilon_t & = 0 \text{ for now}\\
\mathbf{J}  & =  \sum_{t = 0}^{T - 1} (l_{1}(\mathbf{u}_{t}) + l_{2}(\mathbf{x}^* - \mathbf{x}_{t})) \\
\mathbf{x}^{*}  & = f_{inv}(\mathbf{r}) \\
\mathbf{J}  & = \sum_{t = 0}^{T - 1} (l_1(\mathbf{u}_{t}) + l_{2}'(f_{inv}(\mathbf{r}) - \mathbf{x}_{t}))
\end{align}
$$
iLQR algo:
1. Initialisation

start with some guess for control inputs
$$
\{ u_{k} \}_{k}
$$

2. forward: 

simulate system: 
$$
\{ x_{k} \}_k
$$

compute cost function associated with this trajectory

3. backward (quadradic approximation of cost function)

**quadratic approximation of the cost function $l_{2}$ near the current trajectory**

write out one cost function: 

$$
\begin{align}
l(\mathbf{x}_{t})  & =  \lVert \mathbf{r} - f_{fwd}(x_{t}) \rVert ^{2} 
\end{align}
$$



define my network: 

$$
\begin{align}
\mathbf{x}_{t + 1}  & = \mathbf{A}\mathbf{x}_{t} + \mathbf{B}\mathbf{u}_{t} \\
\mathbf{u}_{t}  & = f(\mathbf{u}_{t - 1}, f_{pred}(\mathbf{x}_{t}), \mathbf{r})
\end{align}
$$

$f_{pred}$ is a GRU but we can unroll it to a feedforward neural network, as it only needs one presentation of the input signal. 

the $f$ function can be another simple recurrent network operation. 

we can train $f$ and $\mathbf{B}$, but let's treat $\mathbf{A}$ as given.








