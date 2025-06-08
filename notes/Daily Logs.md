# Logs

Formatting: 
- [ ] = To do
- [?] = In progress (partially completed)
- [x] = Completed

## Day 1: 01/07/2024

Goal: 
- set up basic training skeleton on pycharm.
	- [x] Set up the virtual environment to put in pycharm initially
	- [?] Set up basic training skeleton
		- [?] Deployment within `Network_models` and `Training`
          - In `Network_models`:
            - [?] Base network_model object which extends from the `torch.nn.Module` class
          - In `Training`:
			- [?] Base trainer object (configure as a handle which accepts directly a configuration and outputs training logs as well as stores the trained model.)
	- [x] Familiarise with quaternion operations in Python
		- [x] scipy.spatial.transform.Rotation implementation class for straightforward operations.
		- [x] implementation of the Geodesic distance computation between 
			- [x] two quaternions
			- [x] two rotational matrices
			- [x] two Euler angles (by transforming to quaternions)
		- [x] implement basic quaternion operations in torch.tensor for autograd computation (quaternion multiplications, exponential map)
	
- brush up on group theory and SO(3): Lie group, Lie algebras, and the exponential map. Temporary aim is to know enough to implement stage 1 training of the model (specified rotation either via a rotation matrix (SO(3) member) or a quaternion (unit quaternion)). 
	- [x] Lie group basics
	- [x] Lie algebras basics
	- [x] exponential map basics
	- [x] Quaternion integration
	- [x] Quaternion distance metrics

## Day 2: 02/07/2024

- Monday's	work continued
	- [x] Set up basic training skeleton
		- [x] Deployment within `Network_models` and `Training`
          - In `Network_models`:
            - [x] Base network_model object which extends from the `torch.nn.Module` class
            - [x] Create a simple RNN model that can be configured (input and output settings)
          - In `Training`:
			- [x] Base trainer object (configure as a handle which accepts directly a configuration and outputs training logs as well as stores the trained model.)


## Day 3: 03/07/2024

- Lay down groundworks for visualisation package: 

- [x] Pytorch3D tutorial: visualising meshes.
- [x] Network training pipeline checkup and debugging
	- [x] Redo the file save and load functions
- [x] install PyTorch3D, OpenGL, wxPython (PyTorch3D took up a lot of time)
- [x] initial skeleton APP development (see Day 4 logs for APP structure)
	- [x] Overall menu linking two main functions
		- [x] Visualising network outputs (placeholder only, with control buttons implemented)
		- [x] Behavioural experiment (placeholder only at this stage)
## Day 4: 04/07/2024

- Set up a way to visualise network outputs
	- [x] OpenGL for more general visualisation (check if I can make them work together)
	- [x] Integrate these visualisations into a simple GUI from which I can 
	- for analysis: 
		- [x] input a rotation
		- [x] select a (trained) model
		- [x] see a video of the rotation trajectory
	- for future behavioural experiment
		- [?] have an "experiment" portal that gets me to the behavioural task (two images, forced option). 
- [x] set up GUI

## Day 5: 05/07/2024

- [x] sanity check: compute mean geodesic distance from random predictions. 
- implement alternative loss function calculations, experiment with techniques to speed up training (currently it's fast; but let's try to get it _very_ fast); investigate model performance
	- speed-up
		- [x] first experiment with alternative ways to integrate those quaternions. Keep a measure of time. 
		- [x] experiment with doing that stuff on CPU/CUDA (currently on CUDA, but when I tested it, everything ran a bit faster on cpu). 
	- *conclusion: it seems that my implementation is fast enough*
	- after the speed-up steps, experiment with full-er versions of the loss function
		- [?] silence period loss (output norm loss during silence period to force preparation dynamics onto the nullspace of the angular velocities)
		- [x] geodesic loss (point-wise geodesic loss with $\lambda _{t}$ weighting that increases over time)
		- [?] weight normalisations
		- *temporary result accomplished without the silence period loss and weight norms. Only a linear geodesic gradual loss seemed to be enough*
	- *Bug fix: with 4D inputs, the network struggles with any negative part in the input vector. The temporary fix is to standardize the input value to have both polarities, ordered such that the first polarity is with a positive scalar part, followed by its negation*	

## Day 6: 08/07/2024: result = interactive tuning traces.

- [x] In-depth read (including reference checks) of Jensen et al. 2020 mGPLVM paper. Attempt to intergrate GPLVM method into my repo (say a module called mGPLVM) such that I can call it directly to do a first batch of analysis on my dataset. 
	- [x] Check maths.
	- [x] Check their implementation. The goal is don't use copy and paste but manually implement the method with the help of their [Github](https://github.com/tachukao/mgplvm-pytorch). 
	- [x] Check how they do the visualisations (tuning on a manifold)
	- [x] plan for using it to analyse RNN data: 
		- [x] similar to Kris's paper, start with a ground-truth trajectory of states (target rotations), 
		- [x] extract RNN neural activity to pack into $\mathbf{Y}_{ji}$ ($j$ enumerates the latent space, $i$ enumerates nerons), 
		- [x] perform mGPLVM to get tuning functions and inferred latents.

## Day 7: 09/07/2024

- [x] Write method to construct custom objects (replicating those used in Metzler-type studies)
- [x] Check some literature (using Gemini to provide a basic framework) on how I can implement the `big step up` in the project. 
- [x] Generate training recap across network structures. (added the FC-RNN structure)
	- [x] code a custom function to take a list of network names and generate a 4-panneled plot of results of training different network hyperparameters. 
	- [x] Train models

## Day 8: 10/07/2024

- [x] investigate the use of different learning-rate schedulers (concluded with using a ReduceLROnPlateau scheduler, resetting every $N = 200$ epochs)
- [x] fix the problem involved with the local optimum solution in which the trajectory is pushed to very early in the action period (and exploit the linear weight for geodesic distance). This yields a local optima on strategies such as "spin once and hone", which creates tiers for regularisation loss (hence initial-condition depenedent).
- [x] investigate tuning curves from mGPLVM and Ground Truth (by plotting quaternions)
	- main difference: 
		- in mGPLVM (as well as SO(3)), a rotation is characterised by its _outcome_ ($\pi$ rotation along an axis is roughly equivalent to a $\pi +\epsilon$ degree rotation, which is equivalently captured by a $\pi- \epsilon$ rotation along the negative axis)
		- in trained networks (at least those that are trained well), a rotation is to be accomplished by a linearly annealing loss term on the current effective rotation's geodesic distance to the target rotation. Therefore, the rotation is to be accomplished as fast as possible, and hence the case where a rotation is the same as its conjugate ($\pi$) leads to divergent (even opposite) representations. 
		
- [x] Initial training loop for the CNN-RNN structure

## Day 9: 11/07/2024

Naive end-to-end training of the CNN-RNN structure looks helpless. The model is not learning anything. 
- [x] Investigate transfer learning from both sides (RNN trained on outputting a sequence of rotation from some representation, CNN trained on basic object understanding)
- [x] ConvNeXt addition + pre-trained RNN
- Remark: it works! the model quite successfully learns the task (loss is around 0.5)
- [x] Initial investigations with using mGPLVM to infer tuning information

## Day 10: 12/07/2024

- [x] Built a set of stimuli following the style in Shepard and Metzler paper (elbowed shapes, 5 * 2 (mirrow images))
- [x] Tried training ConvNeXt-RNN on the stimuli as well as applying mGPLVM.

- Remark: _debugged the mistake in training loop (which tends to cause erroneous validation set errors_
- Training results: these abstract images are quite harder to train on than the cow image, perhaps due to their lack of natural prevalence for the pretrained CNN. This is actually the same as in humans encountering the task: these images are _made for_ being unfamiliar.
- Corrected val-set loss around 1.
- tuning curve investigations seem to suggest that the RNN neurons are relatively poorly fit. A lot worse than in the case of the cow-image. 
- For the cow image, tuning curves show SO3 signatures such as being "periodic" (across the poles of this globe you get similar activations). Hints of periodicity in the abstract image, but not as clear. 

## Day 11-12: 15-16/07/2024

[[Meeting with Guillaume 0715]]
From meeting, several steps forward
- [x] Add input noise to the rotation. 
	- [x] constant input noise
	- [x] time-varying input noise (independent Gaussian)
	- [x] silent activity loss?
- [x] crank up weights/weighting function so output is gradual
	- [x] 1. upping the activity loss weight
	- [x] 2. downing the mean $D_{\mathrm{Geod}}$ weighting component; consider exponential weighting function across time
Learn about VAEs 
- [x] read this paper:[Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)
## Day 13: 17/07/2024

Experiment with VAEs on the rotated cows: 
- [x] extend the trainer object for VAEs
- [x] implement and train VAEs
	- [x] experiment with vanilla version first

- [x] read this paper: [[2022 ILQR VAE.pdf]]

## From Day 14 to Day 17: 18, 19, 22, 23/07/2024
[[Meeting with Guillaume 0719]]
- VAE results: successfully implemented a VAE that does a reasonable job at compression. However, this idea (using VAE to reach reduced representation of the data) is paused for now, as if the VAE is only trained on one object just rotated differently, essentially the encoder becomes a rotation inferer and the decoder essentially tries to restore the object 3D model. 
- Residual investigation of pretraining task: 
	- [-] effect of adding noise in thetas directly (instead of quaternions)
		- [?] finding: the network still learns polarised representation in the pretraining task. 
		- [?] the only way to mitigate (really to SO3) is to introduce the closeness between the two poles within features (noisy cross time, so it would be detrimental to the neurons tuned for one pole.)
	- [-] effect of infinite stream of training data
		- [x] implemented a scheme to change up training data given some condition comparing train set performance to val set performance (based on an estimation of local slope) 
		- [?] finding: the network is much better in terms of having a flatter loss landscape across the globe. 
- Investigation of CNN-RNN capabilities
	- proof of concept: can we train a CNN-FC to infer the rotation based on image directly? 
	- currently this seems to be quite lousy. (Maybe because of my implementation of the infinite data stream) 
	- shouldn't be hard at all? 
	- [?] check if turning on the convnet params help (currently they are frozen)
- goal: add a trigger for activating the convnet params within the training loop
- also: add annealing terms to the several components of the loss function aside from geodesics. 

[[ConvDRAW]]


## From Day 18 to Day 20: 24, 25, 26/07/2024
- switch-to-linux turmoil
- set up the SSH branch on Github (on_nonnormal) that contains a lean version of the projec. 
- set up environments
	- Log: lots of turmoils involved with pytorch3d &c. 
	- Correct way to do it: 
		- delete WxPython from requirements
		- pip install requirements including torch, torchvision, torchaudio (currently 2.4.0 installed)
		- conda install pytorch, libtorch, pytorch3d &c. Note: specify builds as the current environment to avoid further difficulties. 
		- conda install wxpython
		- pip install "numpy<2.0.0"

## Day 21: 29/07/2024
Investigation of the latent dynamical system for the forward model to predict. 
- observation 1: use outputs as accelerations typically lead to steady state velocities, which we don't want. 
- temporary fix: 
	1. directly interpret accelerations as velocities - this way allows similarly rich trajectories. 
	2. use additional exponential decay term on the outputs so we force them to converge to 0.
- observation 2: the output matrix limits the space of possible trajectories: 
	- specifically, a random initialisation of the output matrix creates a torus-like structure in the globe. Note, this torus is C-dependent. 
	![[torus_weird.png]]
	![[another network.png]]
- idea: train the C matrix so this is not the case
	- is it possible, though? This seems to involve resonance properties. 

## Day 22: 30/07/2024

Plan 
- [x] investigate if the "limit cycle" phenomenon is the case for using outputs as velocities. 
- [x] investigate angular acceleration and velocity traces for high-amplitude trajectories. 
- [x] look into G's original paper on why this may be the case 
	- [x] then formulate assumption of if this may be fix-able by training C matrix along. 
	- [x] train the relevant matrices and maybe add nonlinearities by class assignment and discrete information maximisation. 

[[Meeting with Guillaume 0730]]


## Day 23: 31/07/2024

Plan: 

investigate several basic things of the system: 
- spectrum of the dynamics matrix
- spectrum of the controllability and observability gramia after the one-step optimisation for C. 
	- implement this as an option in setting up the *DynamicalNetwork*  object. 
	- implement plot spectra as a method under the object.

- investigate using leaked integration. 

set up the training loop for the internal model! 
- Experiment with the following 
	- different layer-ed internal models
		- flat  processor (likely not gonna work)
			- *FlatPredictor*
		- recurrent architectures (fast local dynamics to predict)
			- *RecPredictor*
	- implement my own GRU by the way
	- different encoding functions: 
		- ideal rotation encoder (do this first - proof of concept; inverse task of the pretrain task)
		- PCA encoder (no significant training needed)
		- VAE encoder
			- pretrained and selected
			- co-trained (this may lead to the trivial solution.)
	- curriculum learning on the rotation inference task. 
		- *CurriculumGenerator* (no need, it turns out)

## Day 25: 02/08/2024
- tried initial fitting on several models and specs combinations on quaternions. 
	- observe a three-way relationship between fitabilty, initial condition scalar inverse, and strength of relationship between trajectory magnitude and final rotation magnitude in degrees.
	- such trends are plotted in an obvious folder in the nonnormal subdir on this machine. 
- Note on notations: if there is no specification to any RNN/Linear specs, they are default to: 
- RNN: type = GRU; numLayer = 1; Hidden = 256
- Linear: Hidden = 32

Fill in what I did from 02th to 07th August!

[[Meeting with Guillaume 0807]]

## 08/08/2024: 
1. Profile the processing time for making a dataset (how much is taken to run the dynamics, how much is used for rendering)
- ...

2. Do justice to the cows before moving onto cube objects
- investigate the effects of whitening on training
	- redo the PCA for $\Sigma$ matrix.
- investigate if VAE does any good.

3. Set up the new stimulus: a unit cube mesh with labelled vertices. Figure out the projection step to achieve the following: 
$$
I(q) = \mathrm{Proj}_2(q\mathbf{C}q*) \in \mathbb{R}^{3}
$$
- $\mathbb{R}^{3}$ because we want to use one dimension to specify visibility. 
- Need to specify a torch-based method to go from a bunch of rotations (..., 4) to a bunch of such vertices (..., 8, 3) 

## 12/08/2024: 
- Now the cube-specific training is fully set up, I will spend today investigating different configurations, before moving on to controller design
- what works is a "cheat version" of the task: where the depth information is directly provided - now try extension into 


## 29/08/2024: 

Progress: trained working controllers that successfully performs the target rotation. 
- analysed RDMs for trained networks vs. different representations of rotations
	- [ ] TODO: add in the loop representations based on RPY. 
- plotted dimension-reduced activation curves on different metrics (x, y, z dim in the rotation axis; rotation amplitude)
- [ ] TODO: do the same thing with MDS. (see if the reps using images perform any better)
- [ ] TODO: investigate the actual rotations performed by 
### Notes from Meeting with Guillaume (02/07/2024, 1600-1700)

#### For project

- Vanishing gradient problem in using the distance loss as a delta function. 
	- solution - use point-wise geodesic loss instead (with $\lambda _{t}$ weighting that increases over time), so that all time points come into play.

- Other settings: 
	- silence period activity cost to force recurrent onto the nullspace of the angular velocities (FC readouts) - just an additional norm cost that's on for the silence period
	
- Look into visualisation tools [Pytorch3D]([https://pytorch3d.org/tutorials/render_textured_meshes](https://pytorch3d.org/tutorials/render_textured_meshes)) 
	- for unit-testing quaternion operations
	- as a primer of future work ((series of) images needed for training convnet-RNNs)
	- *for future presentations (make GIFs) - visualising the rotation that the network conducts.*
	
- Once we have the model working (reasonable performance on the pre-train task and extensions, such that we are confident that the rotation is encoded into late-preparation latent activities), we can brainstorm on how to dissect the late-preparation activity as representations of rotations ([[jensen-neurips-2020-manifold.pdf]] as a starting place). 
	- Note: this is encoding for decoding (such that the initial condition can be used to unpack the rotation throughout the action period), hence relevant to actionability inherently. 

#### For exercise
- try to implement my own recurrent cells for full control (forward step & backward step & Adam)
	- Mini-GRU (see iLQR-VAE)
	- Benefits: 
		1. save memory (no need to store training data as a time series).
		2. understand the inner workings of the GRU (and other RNNs) better.

[[Meeting with Guillaume 0719]]
# Notes

## Notes on model designed

README, step 1
- Model input types: 
  - [ ] Rotation matrices (3x3)
  - [ ] Euler angles (3)
  - [ ] Quaternions (4)
	  - [ ] continuous preparation (sustained input for 500 ms)
      - [ ] instantaneous preparation (input at time of movement)
- Model output types: 
  - [ ] Rotation matrices (3x3)
  - [ ] Euler angles (3)
  - [ ] Quaternions (4)
    - [ ] output angular velocities for integration (exponential map)
    - [ ] output modular, discrete rotations per step. 
  