# Full survey of the TEM paper

Significance: 
- this is a model that uses both the attractor framework and incorporation of sensory aff. 
- this can be multivation to what I set up in the controller model which at some point must take in to sensory feedback.
	- though decision point: if the sensory input comes in as prediction error (which can be then transformed to te *action*!) or in full extent
		- in audition, it is widely shown that at least prediction-error type inputs are there, though along with the ground truth. Now, in the simulation model, can take this type of evidence as a motivation to put it out there. 
		- though in full carefulness and following minimal assumptions, we may relay the freedom back to the network (at least don't train a prediction mechanism explicitly)

What I want to know: 
1. implementable details in the method
2. full set of multivations
3. full extent of results so far, critically think how this can be useful in my own work

## Implementation of the TEM

### Generative model

$$
p_{\theta}(\mathbf{x}_{\leq T}, \mathbf{p}_{\leq T}, \mathbf{g}_{\leq T}) = \prod_{t = 1}^{T} p_{\theta}(\mathbf{x}_{t}\mid \mathbf{p}_{t}) p_{\theta}(\mathbf{p}_{t} \mid \mathbf{M}_{t - 1}, \mathbf{g}_{t}) p_{\theta}(\mathbf{g}_{t}\mid \mathbf{g}_{t - 1}, \mathbf{a}_{t})
$$
notations: 
$$
\begin{align}
p(\mathbf{g}_{t} \mid \mathbf{g}_{t -1}, \mathbf{a}_{t})  & : \text{integrating attractor model} \\
p(\mathbf{p}_{t} \mid \mathbf{M}_{t - 1}, \mathbf{g}_{t}) & : \text{place model, with memory (M) and path-int.ed g}  \\
p(\mathbf{x}_{t} \mid \mathbf{p}_{t})  & : \text{observation model}
\end{align}
$$

detailed: 
$$
\begin{align}
\mathbf{g}_{t}  & \sim \mathcal{N}(\cdot \mid \mu = f_{g}(\mathbf{g}_{t - 1} + \mathbf{W}_{a}\mathbf{g}_{t - 1}), \sigma = f_{\sigma_{g}}(\mathbf{g}_{t - 1})) \\
\tilde{g} _{t}  & = \mathbf{W}_{\text{repeat}}f_{\text{down}}(\mathbf{g}_{t}) \\
\mathbf{p}_{t} & \sim \mathcal{N}(\mu = \text{attractor}(\mathbf{g}_{t}, \mathbf{M}_{t - 1}), \sigma = f(\mu))\\
\mathbf{x}_{t}  & \sim \text{Cat}(f_{x}(\mathbf{p}_{t}))
\end{align}
$$

### Inference

$$
q_{\phi}(\mathbf{p}_{\leq T}, \mathbf{g}_{\leq T} \mid \mathbf{x} _{\leq T})=\prod q_{\phi}(\mathbf{g}_{t} \mid \mathbf{x} _{ \leq t}, \mathbf{M}_{t - 1}, \mathbf{g}_{t - 1}, \mathbf{a} _ t)q_{\phi}(\mathbf{p}_{t}\mid \mathbf{x} _{\leq t}, \mathbf{g}_{t})
$$

approximate setup is used (learned conditional distribution) because for the nonlinearities you can't do exact posterior 

so, variational posterior is as follows: 
$$
\begin{align}
\mathbf{x}_{t}^{f}  &  = (1 - \alpha^{f})\mathbf{x}_{t - 1}^{f} + \alpha^{f}\mathbf{x}_{t}^{c}, \quad \mathbf{x}_t^{c} = f_c(\mathbf{x}_{t}) \\
\tilde{\mathbf{x}}_t  & = \mathbf{W}_{\mathrm{tile}}w_{p}f_{n}(\mathbf{x}_{t}^{f}) \\
\mathbf{p}_{t}^{x}  & = \mathrm{attractor}(\tilde{\mathbf{x}}_{t}, \mathbf{M}_{t - 1}) \\
\mathbf{g}_{t} &  \sim q_{\phi}(\mathbf{g}_{t} \mid \mathbf{p}_{t }^{x}, \mathbf{g}_{t - 1}, \mathbf{a}_{t}) \\
\tilde{g}_{t}  & = \mathbf{W}_{\text{repeat}}f_{\text{down}}(\mathbf{g}_{t}) \\
\mathbf{p}_{t} &  \sim \mathcal{N}(\cdot \mid \mu = f_{p} (\tilde{\mathbf{g} }_{t} \cdot \tilde{\mathbf{x}}_{t}), \sigma = f(\tilde{\mathbf{x}}_t, \tilde{\mathbf{g}}_{t})) \\
\mathbf{M}_{t}  & = \text{ hebbian}(\mathbf{M} _{ t - 1}, \mathbf{p} _{ t}) \\
\to  & \mathbf{x}_{t + 1} \to \mathbf{g}_{t + 1} \to \mathbf{p}_{t + 1}
\end{align}
$$

step 1: compress one-hot sensorium into two-hot (for easy computation) and filter (approximates how the neuron may exponentially filter information tmeporally). 
step 2: $f_{n}(\cdot)$ demeans, ReLU, and unit normalizes the representation. $w_{p}$ is the scalar weight and $\mathbf{W}_{\mathrm{tile}}$ is the tiling matrix to  project the state input into hippocampal dimensions. 
step 3: 
$$
q_{\phi}(\mathbf{g}_{t} \mid \mathbf{x}_{\leq t}, \mathbf{M}_{t - 1}, \mathbf{g}_{t - 1}, \mathbf{a}_{t}) = q_{\phi}(\mathbf{g}_{t} \mid \mathbf{g}_{t - 1}, \mathbf{a}_{t}) q_{\phi} (\mathbf{g}_{t} \mid \mathbf{x}_{\leq t}, \mathbf{M} _{ t - 1})
$$

**note: path-integrates and integrates with sensorium, probabilitically**
- if we care about means only, what we end up here is a weighted combination of means inferred from each source. 
- now, this indeed may come with assumptions for that we *can* just linearly combine these representations. is it straightforward considering a mix on the attractor (neural) manifold?

indeed: both distributions (at least approximated as here) are Gaussian and have MLP-parametrized mean and precision --> precision-weighted mean as a result. 

now: $p_{t}^{x}$ are not random variables - albeit that they should learn similar rep as $p^{t}$. These are though effectively the output of a function: what's the memory rep given my sensory evidence? 
- *this though can be extended with probabilities: a distribution in the external variable --> a distribution on the rep manifold*

step 4: $f_{\mathrm{down}}$ to down-sample the $\mathbf{g}_{t}$ (as an information bottle-neck), and re-up the dimenssionality by a fixed $\mathbf{W}_{\mathrm{repeat}}$. 
- element-wise multiplication, $\tilde{\mathbf{x}}_{t} \otimes \tilde{\mathbf{g}}_{t}$ 
- *the two fixed weight matrices are designed such that the element-wise multiplication yields an outer product followed by vector-reshape*
![[Pasted image 20250126155901.png]]
### Memories

$$
\begin{align}
\mathbf{M}_{t}  & = \lambda M_{t - 1} + \eta(\mathbf{p}_{t} - \hat{\mathbf{p}}_{ t })(\mathbf{p}_{t} + \hat{\mathbf{p}}_{t})^{\top}
\end{align}
$$
(also possible to have a separate inference based matrix)
$$
\mathbf{M}_{t}^{x}  = \lambda \mathbf{M}_{t - 1} ^{ x} + \eta (\mathbf{p}_{t} - \mathbf{p} _{ t}^{x}) (\mathbf{p}_{t} + \mathbf{p} _{t}^{x})^{\top})
$$

$\mathbf{p}_{t}^{x}$ differes from $\hat{\mathbf{p}}_{t}$ by that it is the inferred hippocampal activation from the snsorium as the input to the attractor. 

### The attractor

$$
\mathbf{h}_{\tau} = f_{p}(\kappa \mathbf{h}_{\tau - 1} + \mathbf{M}_{t - 1}\mathbf{h}_{t - 1})
$$

Here, $\mathbf{h}_{0}$ can be either $\mathbf{g}_{t}$ or $\mathbf{x}_{t}$ depending on if we are doing generative or inference. 

## Summary

1. the environments are graph structures on which the agent wanders around, node to node, observing sensory objects at each point. *goal is to predict sensory experiences at each time step*
2. actions are provided to the agent; not an RL problem but one of sensory prediction
3. model is not given location information (nor trained upon it)
4. *sensory input provides no spatial information*; here is 1-hot vector given the object experienced
5. network = *path integrator* + *memory network* (combine path integrated and sensory information)
6. $\mathbf{p}$ are "grounded locations"; $\mathbf{g}$ "abstract"
7. the network must learn to understand sensory predictions in a meaningful way. not obvious from a one-hot vector! (need to think hard on how this can be respected in my case, where in a sense the sensory rep *is* a parametrization of the manifold; must separate underlying manifold and sensorium, which is not straightforward)
8. network weights are learned through runs in the environment; hebbian weights are adjusted online to construct association between sensorium and structural knowledge. 


