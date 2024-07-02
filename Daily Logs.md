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

### Notes from Meeting with Guillaume (1600-1700)

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
  