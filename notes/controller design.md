## Controller design: 

general workflow: 

$$
g_{t} \to y_t \to I_t, \hat{g}_{t + 1} \to \hat{I}_{t + 1} \to g_{t + 1}
$$

1. (*learn to*) path-integrate given plant activity ($\hat{g}_{t + 1} = f_p(g_{t}, y_{t})$)
	- $f_{p}(g_{t}, y_{t}) = g_{t}(f_{ya}(y_{t}))$; ground truth here exists, obviously, as there is a fixed readout model that casts actions to the sensory environment which is supposed to be aligned with the internal representation
2. (*learn to*) align with sensory inputs (internal model) ($\hat{I}_{t} = f_a(g_{t})$)
	- $dI_{t} = d\phi_I(\theta_t) = \frac{d\phi_{I}}{d\theta} \frac{d\theta}{dt}$
	- $dg_{t} = \frac{dg}{d\theta} \frac{d\theta}{dt}$
	- $I_{t} = \phi_{I}(f^{-1}_{\mathrm{encoding}}(g_{t}))$
	- though we don't require the image to be a unique mapping from the rotational configuration (multiple configuration can give the same sensory evidence)
3. (*learn to*) resolve misalignments between sensory inputs ($g_{t} = f(\hat{g}, I_{t}, \hat{I}_{t + 1})$) 
	- $g_{t} = f_{p}(\hat{g}_{t}, a_{c}), a_{c} = f_c(I_{t}, g_t)$ 
		- note: $f_c$ could be $f_c'(I_{t}, \hat{I}_t = f_a(g_t))$. However, this thing is not necessarily unique (pair of prediction and sensory input can be ambiguous in terms of the implied corrective action.)
		- strategies: 
			- to train $f_c$ without thinking about prediction necessarily
	- or to do as in TEM: $P(g_t) = P(\hat{g_t})P( \tilde{g}_t),\quad P(\tilde{g}_t) \sim \mathcal{N}([\mu, \sigma] = f(M_{t - 1}, I_{\leq t})$
		- in that case we also want to revise the CAN thing to also incorporate a sense of uncertainty. not necessarily easy? 

all the alignments seem to be very trainable during random exploration. 
- for example, self-supervise on implied actions: coevolve the action readout matrix with the implied action matrix, by aligning: 
	- $\hat{g}_t = f_p(g_{t - 1}, f_{ya}(y_t))$ (note that $f_p$ is fixed update rule on the CAN)
	- $f_{ya}(y_t) = f_{c}(I_{t}, g_{t - 1})$ (this will work if $g_{t - 1}$ relatively successfully tracks internal state)
	- *explicit alignment might not need to happen?* all it takes is to successfully do the action inference given the plant state. we may additionally constrain $a_c$ to be small, in order to push $I_t$ close to $g_t$

these may happen along with descent on the policy, such that we constrain the exploration to end where we are close to the null state. 
- train on the final distance to the null state

a few extra things to consider: 
- certainty of the internal model
- certainty of the environmental evidence 

module 1: 
CAN (to hold $g_t$; has an internal way to comprehend actions (so $f_p$ is implemented with $g_t(a_t)$ or CAN.step(a_t)))
$$
\begin{align}
\tilde{ g_{t}} &  = f_{p}(g_{t - 1}, f_{ya}(y_t)) = g_{t - 1}(a = \hat{a}_t = f_{ya}(f_{gu}(g_{t - 1}))) \\
g_{t} & = f_{c}(I_{t}, \tilde{g}_{t}) = \tilde{g}_{t}(a = a_{c} = f_{c}(I_{t}, \tilde{g}_{t}))

\end{align}
$$

module 2: 
a policy function to train based on the CAN, which gives inputs to the plant
$$
u_t = f_{gu} (g_t)
$$

module 3: 
the plant to be controlled (ISN + random readout)
- initialization from baseline. 
- a fixed, randomly initialized readout to act on the environment. 
$$
\begin{align}
y_{t}  & = Ay_{t - 1} + Bu_t + \epsilon_a \\
a^\star_{t}  &  = Cy_{t} + \epsilon_c
\end{align}
$$

module 4: 
optional image comprehension module (can train a priori for now, based on our CAN-training paradigm, using the observed image as the parametrization (so, image similarity properties are learned!))

$$
I_{t+1} = I_{t}(a^{\star}_{t})
$$
update of the image based on the actual action taken

