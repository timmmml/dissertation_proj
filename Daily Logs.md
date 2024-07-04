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

- [ ] sanity check: compute mean geodesic distance from random predictions. 
- implement alternative loss function calculations, experiment with techniques to speed up training (currently it's fast; but let's try to get it _very_ fast); investigate model performance
	- speed-up
		- [ ] first experiment with alternative ways to integrate those quaternions. Keep a measure of time. 
		- [ ] experiment with doing that stuff on CPU/CUDA (currently on CUDA, but when I tested it, everything ran a bit faster on cpu). 
	- after the speed-up steps, experiment with full-er versions of the loss function
		- [ ] silence period loss (output norm loss during silence period to force preparation dynamics onto the nullspace of the angular velocities)
		- [ ] geodesic loss (point-wise geodesic loss with $\lambda _{t}$ weighting that increases over time)
		- [ ] weight normalisations
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
  