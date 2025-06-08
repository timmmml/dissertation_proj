Theme: predictive coding

- predictions: essential for decisions, actions, and controls, by being able to predict the consequence of our actions
- this is a special case of inference: using known quantities to infer unknown quantities
	- (past --> future being one special case)

but the above is not *predictive coding*
- *predictive coding* is a particular way to use neural architecture to solve predictions
	- there are other ways that does prediction that has nothing to do with predictive coding

"Predictive coding and Bayesian Inference in the Brain"

- technical definition: specific neural responses somewhere in the brain that *represents the prediction error*
- if this is computed, may be the case that no neuron explicitly codes for it. (may be implicitly represented in cellular or subcellular processes)
- **HERE, WE SPECIFICALLY TEST FOR EXPLICITLY REPRESENTED, NEURAL OUTPUT LEVEL PREDICTIVE CODING**

well-known example: prediction error in the DA circuit
- temporal difference rule (Dayan and Abbott 2001 textbok)
- finding is in Schultz, Dayan, Montague 1997

importance: this is probably *the most* experimentally tested principled theory in computational neuroscience. 
- this coding is explicitly beneficial for the computation

here, the prediction is scalar (simple).  

But, the predictive coding talked about today is that of high-dimensional inputs (high dimensional predictions). 
- harder to test experimentally

to make this link, we first refer back to the Bayesian brain. 
- idea: the brain implements some internal model that *reflects statistical regularities of the outside world* ($P_{\theta}(\mathbf{y})$, along with learned $P_{*}(\mathbf{x}\mid\mathbf{y})$ to predict $P_{\theta}(\mathbf{y}\mid\mathbf{x})$)
- here, we want to see how neural circuit dynamics implement and code for these Bayesian posteriors
- within this field, we have a family of model under the name "Bayesian Predictive Coding"

Predictive coding is, among many, one way to compute the posteriors. 

A research project here: 
- neural model
![[hierarchical model.png]]
in here, everything is linear and Gaussian but we ignore the shortcomings for now. 

sensory inputs here are $\{ x_{i} \}_{i}$ (let them all be vectors)
- define that the statistical dependencies here follow a tree structure. 

- here we also say that any covariance structure is captured by the layer above.

Now, we want to invert the process: we observe $\mathbf{x}$ only 

- under Bayesian brain hypothesis we care about the inversion process

One particular algorithm for this parameter inference (and latent factor recovery):
- EM - the goal is to retrieve a posterior distirbution over all ys

let's make the cruel assumption to approximate the true posterior by a delta distribution. 
- we will optimize the KL wrt. parameters $\bar{y}$ (in this case we only need these) against the true posterior over $y$. (zero temperature EM - very bad idea in general)

- the particular gradient 

![[zerotemperature EM.png]]
