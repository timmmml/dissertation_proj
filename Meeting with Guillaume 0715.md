# Priorities

- **Project recap**
- **Project further directions and outlook**
- *some macro questions I had*
- *PhD talks if have time*

# Project recap

## Pre-training task

Loss function: $$\lambda_{1}\sum_{t= T_a}^{T}\lVert \boldsymbol{\omega_t} \rVert_{2} + \lambda_{2} f(\boldsymbol{\omega}, q^\star)$$

where 
$$
\begin{align}
f(\boldsymbol{\omega}, q^\star)  & = \sum_{t = T_a}^{T} w_tD_{\mathrm{Geod}}(q_t(\boldsymbol{\omega}), q^{\star}) \\
\text{where,} \\
D_{\mathrm{Geod}}(q, q') & = 2\arccos(q \cdot q') \\
\text{and} \\
q_t(\boldsymbol{\omega})  & = \prod_{\tau = T_a}^{t}\mathrm{Exp}_q(\omega_\tau\delta t)\otimes q_0 \textit{, (strictly left multiplies)}
\end{align}
$$

hyperparameters for plots: 

$$
\begin{align}
\lambda_{1}  & = 1 \\
\lambda_{2}  & = 1 \\
\delta t  & = 1 \\
lr  & = 0.01, \overset{a}{s.t.} \mathrm{ReduceLROnPlateau, LB = 0.0005}\\
\end{align}
$$
$a$: small caveat: the scheduler is set to reset every n epochs, until $e = \frac{3}{4}E$ is reached. After that, no more reset and the scheduler is alowed to hone in on the minimum. 

Investigated three simple $f_w(t)$'s to generate $w_t$ for the distance loss

$$
\begin{align}
f_{w1}(t)  & = \delta(t - T)\\ 
f_{w2}(t)  & = \frac{t - T_a + 1}{T - T_a} \\
f_{w3}(t)  & = 0.5 + \frac{t - T_a + 1}{T - T_a}
\end{align}
$$

The first one is simply using the final distance loss only. Vanishing gradients; doesn't work. 

For comparison between the second and the third, it wouldn't be straightforward from best networks but one needs to consider the loss landscape. Basically, $f_{w_{2}}$ seems to create this valley in which the network pushes some trajectory to the start of the action phase, exploiting the very $w_{2}(t)$. This renders further optimisation on rotational paths difficult (can be seen in that the regularisation loss wouldn't decrease even though there is a clear global minimum). 

Example (even with $f_{w3}$ - it's only that this loss function is better at helping networks converging to the *global* minimum): this is 100 epochs of training 5 splits on the same training set. 

![[Example local minimum.png]]

The below plot show comparisons between network structures under $f_{w_{3}}$ only. 

![[pretrain_summary.png]]

As one can see, performance scales with parameters. 

Example neural activity traces: 

Hidden layer: 
![[FC_32_GRU_1layer_8hidden_hidden_activations.png]]

Output layer: 
![[FC_32_GRU_1layer_8hidden_output_activations.png]]

We can see that the network learns the task and chooses to do the rotation very fast. 

Tuning curves: (the ball should be read as follows: each position represents a rotation: axis = axis of rotation, distance to origin = angle of rotation; a colored point represents neuron (hidden unit) activity: blue is low, red is high)

Example 1: GRU(8) no FC

Ground truth:
![[GRU8_best_model_tuning_curves_Ground_Truth.png]]

mGPLVM predicted:
![[GRU8_best_model_tuning_curves_mgplvm.png]]

Example 2: GRU(8) FC(32)

Ground truth:
![[GRU8_FC32_best_model_tuning_curves_Ground_Truth.png]]

mGPLVM predicted:

![[GRU8_FC32_best_model_tuning_curves_mgplvm.png]]

In both cases, we see one discrepancy: 
- model predicts "periodicity", where 180 degrees to one direction is represented similarly as 180 degrees to the other direction. 
- the periodicity is not there for ground truth - instead, GRU units learn directed rotation trajectories and hence the periodicity is not here. 

Remark: 
- This is due to how the task is set up: we impose the distance to ideal rotation on every timestep, and hence network is forced to make the most efficient rotation.
- An optimal rotation trajectory is unique (except for 180 degree exact cases); a rotation is not. 

## CNN-plugged GRU

First investigated training a fresh CNN-RNN architecture. Not much hope. 

Migrated to transfer learning
- for CNN, use pretrained weights from torchvision
	- **ConvNeXt Tiny**
	- ... try more?
- for RNN, use any pretrained model

In between, add a MLP to reinterpret CNN output. The MLP output should match RNN input. 
- this is why I included the FC in the previous section. These are to introduce the RNN with a higher-dimensioned input instead of limiting to 8. 

training: freeze CNN weights and output weights. Train RNN and MLP. 

Training stimuli:

cow: 

![[Ideal_1.gif]]

abstract shapes (c.f. Shepard & Metzler paper):

![[Ideal_36.gif]]

Fitting these take a longer period. 

preliminary: 
- tried fitting for the FC16-GRU8 case (because it's simplest)
- performance scales with input resolution for cows, not much so for abstract shapes.

comparison: first thing - effect of image resolution: 
![[cnn_summary.png]]

- Actually not much differnce (except with A1 stim - looks a bit spurious?)

Example activity traces: 

Hidden:
![[ConvNeXt_Tiny_FC_16_GRU_8_A1_5x6x5_centered_1.1_res256_hidden_activations.png]]

Output:
![[ConvNeXt_Tiny_FC_16_GRU_8_A1_5x6x5_centered_1.1_res256_output_activations.png]]

Tuning curves: 
Example 1: Cow res 256
Ground Truth:

![[ConvNeXt_Tiny_FC_16_GRU_8_cow_res256_best_model_tuning_curves_Ground_Truth.png]]

mGPLVM predicted:

![[ConvNeXt_Tiny_FC_16_GRU_8_cow_res256_best_model_tuning_curves_mgplvm.png]]

Example 2: A1 Res 256 
Ground Truth: 

![[ConvNeXt_Tiny_FC_16_GRU_8_A1_res256_best_model_tuning_curves_Ground_Truth.png]]

mGPLVM predicted:
ConvNeXt_Tiny_FC_16_GRU_8_A1_res256_

![[ConvNeXt_Tiny_FC_16_GRU_8_A1_res256_best_model_tuning_curves_mgplvm.png]]

# Further directions

potential ways into this week: 
- think about how exactly CNN-RNN structures are to be trained to represent rotations for more abstract objects. What should be the goal here? 
	- abstract objects are made for being confusing rotation-wise (if we replace the image in the mental rotations task with cows then wouldn't need to mentally rotate for comparison. The thing mustn't have a ready representation)
	- should generalise to zero-shot rotation predictions or at least checking. Pair-images as input?
	- roll-outs? 

- representations: 
	- now I plot tuning curves, and see some qualitative differences. 
	- How can we perhaps quantify this? We should be dealing with similar types of data when we have better networks.
		- variance analyses of these as basis sets (re-visual coding - but here trajectories are simple. Maybe experiment with more complex trajectories?)
		- some way to compare representations across networks? Then can quantify to what extent representations change from what loss manipulations. 
	- more loss functions to make this biologically plausible (morph in some sparsity constraints/information efficiency constraints?)

- macro stuff moving forward
	- how do we begin to think about coding in mixed spaces (Rotations + Translations, across time)

(where should we claim victory on this project)


# Macro/PhD questions
d computational neuroscience (with focus on basic science questions, but also cognitively-enriched BCI applications).
kk
My goal: study toward computational neuroscience (with focus on basic science questions, but also cognitively-enriched BCI applications).

- topic 1: PhD in CBL (I know bits about the application here already, but question on collaboration among labs); Gatsby?
- topic 2: US PhD on computational neuroscience/computer science labs 
- a second MS degree?

Path forward:
- me vs. a physics student 
	- what maths am I missing? 
	- the best way is to learn through projects such as the present one
		- so far this has been a lot of software engineering but not much maths
		- must be a lot in the next steps; what should i read to get started? 
- what should be the skills i focus on developing as a computational neuroscientist? 
- brief words on course selection

[[Course selection]]

## note 
- input cnn encoder + output cnn decoder
- parse visual fb /// predict visual consequence

pretrain autoencoder with CNN to compress representaiton of the input in Euclidean space. (for single objects)
- then work in the space of the compressed representation. 

image -> encoder (pretrained) -> FC -> network 
network -> FC -> decoder (pretrained) -> image

objective for pretraining: world model
- sample random initial state (N) (figure out spread for rotations to span )
- co-evolve the prediction of visual consequence / RNN motor sequence

