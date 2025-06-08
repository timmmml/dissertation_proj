# Outline

- Recap and update progress on the mental rotations project
	- how should I talk about findings here?
- Discuss and define the new dissertation project
	- how to define actionable fruits to reach at?  
- *Discuss a bit on the side project I have with Marcelo (RT-RNN)*


## Recap and Updates

Recap (very honest and chronological): 
- We were interested in how a neural circuit codes for a motor plan in SO(3), and investigated by training networks subject to various conditions
- We initially trained GRUs on quaternions for the target rotations. 
	- we found that we must present the quaternions twice for the network to solve the task nontrivially
	- *I later found that it suffices to just regularise the quaternion, too*
	- *in fact, same for just random quaternions*
		- this is actually quite interesting.
	- *I later found that the cubes can also train the network*
- We then wanted to generalise to images, which was a bit more challenging
	- I used pretrained models (ConvNeXt2) concatenated with pretrained quaternion-based networks, which doesn't really work. 
	- we thought about using VAE to find a lower feature space that we operate in (instead of pixel space)
- about that moment we diverted to using controller-predictor architectures, such that we want to control an ISN to some initial state such that the automatic rollout, up to a decoding matrix, corresponds to an appropriate time series of angular accelerations that lead to the desired rotation. 
	- we discarded the cows for cubes at this moment
	- we trained controllers that directly set the latent states or have their own latent states and inject inputs

- We found that, using representation similarity analysis and visual inspection, these networks generally code for the target rotation in the axis-angle form, in that a population code usually defines the axis of the rotation, and strength along that code codes for the extent. 
- In the first PCs, there looks like there is a threshold at which neural representations changes abruptly.

- I then investigated a bit generalising Will Dorrell's work on actionable representations. I was able to tackle the problem, basically, deriving myself the matrix operation needed to transfer wigner-d matrices into real wigner-d matrices. 
- It comes down to training a linear recombination matrix that we apply to the vanilla irrep. It's just that in this case irreps can be any column (not only the mid ones - which are spherical harmonics) of the wigner d, and evaluates any rotation. 
- as a result of grapples, I found some beautiful grid-like structures

To what extent these grids are relevant I have no idea, though I have a hunch that they may appear if we constrain the hierarchical network to form a mid-layer representation of the rotation target that *is updated every step of the output network rollout*, such that actionability can be sort of helpful. 

- how should i talk about this work in statements of purpose? 
	- I feel like there is no clear-cut results to present; only qualitative ones.
	- currently i put it as a part of my dissertation project, in which we set up the games for investigating hierarchical motor planning in neural circuits. 
- How should we then talk about results in the next project? 

## How does the brain *hierarchically* represent 3D rotations *in different actuators*, *across different tasks*? 

- (title can be subject to change, though I quite like this idea of investigating hierarchical reps)
- *I am quite open to new ideas too in case we couldn't find clear milestones for this one*

action plans as follows (very vague): 
- context: it is commonly assumed that upstream areas to M1 represent higher-order information, for example a *sequence* of modularised actions. 

	- Grafton and Hamilton 2007 (review; repetition suppression)
	- Ito and Murray 2023 (NN; fMRI, alignments between regions across tasks; graded alignment along association -> motor axis, "rich training" of deep nn recovers this hierarchy)
- context 2: it has been shown that RNNs can self-organize motor hierarchies 
	- Han et al. 2020 (multi-timescale, stochastic RNN for RL; learns subcgoals and develop hierarchy with internal dynamics; faster adaptation to a recomposed new task with previous subgoals)
- context 3: modular architecture has been associated with success in ML (Mittal, Bengio, Lajoie 2022)

and Michaels et al. 2020 etc. 

- We can address this self-organisation of hierarchical computations, leveraging our existing queries into SO(3), in that we now know what reps for goals and reps for actions look like (kind of different!)
- goal: compare a few conditions to see when does brain-like hierarchy arise
	- *I want to be able to use this structure for better motor decoding - what are your thoughts?*

decisions: 
- should we implement multiple-rotation tasks to see if "SMA" codes for rot sequences, or should we focused on one rotation and see if "SMA" codes for rot targets? 

conditions: 
- (sequential vs. singular rotatations) x (similarly vs. differently constrained actuators) x (one vs. many rotated objects) x (one vs. two cognitive tasks)

how can we scope the efforts? and how do we quantify the results in a publishable/inviting for future data manner? 

## My third ref letter and project with Marcelo

- ...