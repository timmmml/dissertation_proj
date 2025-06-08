# Part 1: data-driven circuit models, concensus visual areas dynamics in mice
- Guillaume's work with Javadzadeh, Schimel, Hofer, and Ahmadian

 for modelling cortical dynamics: 
- **minimal circuit models**: N-area population models of EI balance/attractor models 
	- coarse spiking stats
	- stim tuning, oscillations/modes
	- *explain a lot with little - qualitative models amenable for thoeretical analysis*; semi-mechanistic
	- simple models --> testable predictions
	- *but not even trying to assimilate data for quantitative stuff*
- **LFADs-based models (abstract dynamical models)**: not even meant as "mechanistic models" but to capture statistical latents
	- Pandarinath et al. 2018, Genkin et al. 2021, Schimel et al. 2022
	- good to do single-trial inference &c.

Why not use the data approach on minimal circuit models?
- blurring the boundary between the dichotomy

Q form Sasha: definition of "minimal circuits"? 
- it comes from circuit constraints and manipulation-abilities (E-I partitioning &c.) (hypothesis-based explanability)
	- additionally, biological plausibility and insights to the mechanism

## multi-area cortical dynamics

- Javadzadeh and Hofer, 2022
- GNG task, V1/LM multiarea single-trial recordings ($n \approx 30$ per area per mice)
	- also optogenetics  - chR2 PV cells to photoinhibit LM/V1 (by exciting inhibitory cells s.t. the rest of the net is inhibited)
- fit latent circuit models to this data

- iLQR-VAE (ICLR talk)
	- model data as originating fom some *driven* LDS characterised by
$$
\begin{align}
\dot{x}  & = f_{\theta}[x(t), u(t)] \\
u(t)  & \sim p_{\theta}(\cdot)
\end{align}
$$
- temporal fluctuation of firing rates originate from latent trajectories lead to observations
	- optimize the marginal likelihood 

from top to down: 
infer control variables, which are integrated by the dynamical system to latent variables, which are then mapped on to firing rates and then observations


With VAE we can maximize the ELBO loss here: 
$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_{\phi}(u)}[\log p_{\theta}(\text{spikes}|\mathbf{u})] - \mathcal{KL}(q_{\phi}(\mathbf{u})| | p_{\theta}(\mathbf{u}))
$$
here we can learn dynamics defined by $\theta$ and infer $\mathbf{u}$
(essentially a LFADS-like approach)

- generative model and recognition are learned 

but: 
- exploit control/inference duality to find MAP as a control problem
- control the system to good trajectories for the dat
$$
\log p_{\theta}(\mathbf{u}|\text{spikes}) = \sum_{t}\log p_{\theta}(s_{t}|x_{t}(\mathbf{u})) + \log p _{\theta}(u_{t})
$$

use diff. iLQR to do inference

latent model, but now mechanistically interpretatle

$$
\tau \dot{x} = -x + W \phi[x] + Bu
$$
- no "bias" in the latent dynamics (if included, the model will learn a high bias)
($\phi[x]$ is the single-neuron firing nonlinearity)

- granting the ability for $u(t)$: almost a "must"

Constraints:
1. latents are made of local EI structures
	1. local EI
	2. long term E only
2. cook these as C matrix constraints 
	$$
	r(t) = \exp (Cx + d)
	$$
	- bias here (d)
	1. structured sparsity penalty on the readout matrix - such that each neuron is connected to either the E population or the I population (C elements are positive)

3. constraints on the input channel? 
	1. we know the task structure - stim nature and timing --> use this to structure trial-specific *priors* (prior variance is "something small" in input-silent stage, "something large" in input stage) - input types are used to do specify (learn) differnet priors
		- inputs are independent over time (temporal smoothness in recurrent dynamics)
	1. iLQR-VAE to infer the specifics; 
	2. do the same for optogenetically controlled trials (this is known).

- we can by the way use pv cell labelling to filter models.


What do the dyanmics look like?
- query the autonomous flow-field
	- approximate line attractor (based on the slowest and second-slowest eigenvalues of the jacobians)
	- line attractor only exist in (biologically) constrained models; 
	- no line attractor without long-range connections. 
	- *use transient input to push the network onto somewhere on the line attractor*

- *simple* understanding of linear attractors from long-range connections 


# CEBRA
